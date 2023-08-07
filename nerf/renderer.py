import os
import math
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
import nvdiffrast.torch as dr

import mcubes
import raymarching
from .utils import custom_meshgrid, safe_normalize
import logging
from activation import trunc_exp, biased_softplus


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                       self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

    def reset_parameters(self):
        @torch.no_grad()
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(weight_init)


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

@torch.cuda.amp.autocast(enabled=False)
def near_far_from_bound(rays_o, rays_d, bound, type='cube', min_near=0.05):
    # rays: [B, N, 3], [B, N, 3]
    # bound: int, radius for ball or half-edge-length for cube
    # return near [B, N, 1], far [B, N, 1]

    radius = rays_o.norm(dim=-1, keepdim=True)

    if type == 'sphere':
        near = radius - bound # [B, N, 1]
        far = radius + bound

    elif type == 'cube':
        tmin = (-bound - rays_o) / (rays_d + 1e-15) # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 1e9
        far[mask] = 1e9
        # restrict near to a minimal value
        near = torch.clamp(near, min=min_near)

    return near, far


def plot_pointcloud(pc, color=None):
    import trimesh
    # pc: [N, 3]
    # color: [N, 3/4]
    logger.info('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class DMTet:
    def __init__(self, device='cuda'):
        self.device = device
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [1,  0,  2, -1, -1, -1],
            [4,  0,  3, -1, -1, -1],
            [1,  4,  2,  1,  3,  4],
            [3,  1,  5, -1, -1, -1],
            [2,  3,  0,  2,  5,  3],
            [1,  4,  0,  1,  5,  4],
            [4,  2,  5, -1, -1, -1],
            [4,  5,  2, -1, -1, -1],
            [4,  1,  0,  4,  5,  1],
            [3,  2,  0,  3,  5,  2],
            [1,  3,  5, -1, -1, -1],
            [4,  1,  2,  4,  3,  1],
            [3,  0,  4, -1, -1, -1],
            [2,  0,  1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device=self.device)

        self.num_triangles_table = torch.tensor(
            [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long, device=self.device)
        self.base_tet_edges = torch.tensor(
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=self.device)

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)

        return torch.stack([a, b], -1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N,
                           dtype=torch.float32, device=self.device),
            torch.linspace(0, 1 - (1 / N), N,
                           dtype=torch.float32, device=self.device),
        )  # indexing='ij')

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x, tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x, tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim=-1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4, return_uv=True):
        # pos_nx3: [N, 3]
        # sdf_n:   [N]
        # tet_fx4: [F, 4]

        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)  # [F,]

            # a valid tets not all positive (out space) and not all negative (inner)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,
                                            self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(
                all_edges, dim=0, return_inverse=True)

            # find out the edges across the surface to interpolate and refine
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)
                               ].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones(
                (unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=self.device)
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1
        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        
        # interpolate edges by sdf
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(
            4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)

        if return_uv:
            # Get global face index (static, does not depend on topology)
            num_tets = tet_fx4.shape[0]
            tet_gidx = torch.arange(num_tets, dtype=torch.long, device=self.device)[
                valid_tets]
            face_gidx = torch.cat((
                tet_gidx[num_triangles == 1]*2,
                torch.stack(
                    (tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
            ), dim=0)

            uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)
        else:
            uvs, uv_idx = None, None
        return verts, faces, uvs, uv_idx

###############################################################################
# Regularizer
###############################################################################


def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
        torch.nn.functional.binary_cross_entropy_with_logits(
            sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################


class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, tet_mlp, opt, device='cuda'):
        super(DMTetGeometry, self).__init__()

        self.opt = opt
        self.device = device
        self.tet_scale = torch.ones(3, device=device)
        self.grid_res = grid_res
        self.marching_tets = DMTet()

        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        # for 64/128, [N=36562/277410, 3], in [-0.5, 0.5]^3
        self.verts = torch.tensor(
            tets['vertices'], dtype=torch.float32, device=self.device) * 2
        # for 64/128, [M=192492/1524684, 4], vert indices for each tetrahetron
        self.indices = torch.tensor(
            tets['indices'], dtype=torch.long, device=self.device)
        self.generate_edges()

        self.tet_mlp = tet_mlp
        if tet_mlp:
            self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3)
            self.encoder = self.encoder.to(device)
            self.mlp = MLP(self.in_dim, 4, 32, 3, False).to(device)
            self.sdf = None 
        else:
            sdf = torch.nn.Parameter(torch.zeros_like(
                self.verts[..., 0]), requires_grad=True)
            self.register_parameter('sdf', sdf)
            deform = torch.nn.Parameter(
                torch.zeros_like(self.verts), requires_grad=True)
            self.register_parameter('deform', deform)

        if opt.base_mesh and os.path.exists(opt.base_mesh):
            self.init_tet_from_mesh(opt.base_mesh)

    def reset_tet(self, reset_scale=True):
        if self.tet_mlp:
            self.mlp.reset_parameters()
        else:
            self.sdf.data = torch.zeros_like(self.verts[..., 0])
            self.deform.data = torch.zeros_like(self.verts)
        if reset_scale:
            self.reset_tet_scale()

    def get_sdf_from_mesh(self, base_mesh):
        logger.info(f'[INFO] init sdf from base mesh: {base_mesh}')

        import cubvh
        import trimesh
        mesh = trimesh.load(base_mesh, force='mesh')

        scale = 1.5 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
        center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
        mesh.vertices = (mesh.vertices - center) * scale

        # build with numpy.ndarray/torch.Tensor
        BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
        sdf, face_id, _ = BVH.signed_distance(
            self.verts, return_uvw=False, mode='watertight')
        sdf *= -1  # INNER is POSITIVE
        return sdf

    def init_tet_from_mesh(self, base_mesh):
        sdf = self.get_sdf_from_mesh(base_mesh)
        self.init_tet_from_sdf(sdf)
        # visualize
        # sdf_np_gt = sdf.cpu().numpy()
        # sdf_np = self.mlp(self.encoder(self.verts)).detach().cpu().numpy()[..., 0]
        # verts_np = self.verts.cpu().numpy()
        # color = np.zeros_like(verts_np)
        # color[sdf_np < 0] = [1, 0, 0]
        # color[sdf_np > 0] = [0, 0, 1]
        # color = (color * 255).astype(np.uint8)
        # pc = trimesh.PointCloud(verts_np, color)
        # axes = trimesh.creation.axis(axis_length=4)
        # box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
        # trimesh.Scene([mesh, pc, axes, box]).show()

    def init_tet_from_sdf(self, sdf, pretrain_iters=5000, lr=1e-3):
        if self.tet_mlp:
            self.mlp.reset_parameters()
            # pretraining
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)
            
            #batch_size = min(10240, self.verts.shape[0])
            batch_size = self.verts.shape[0]
            pbar = tqdm(range(pretrain_iters), desc="init dmtet mlp from sdf")
            for i in pbar:
                rand_idx = torch.randint(0, self.verts.shape[0], (batch_size,))
                p = self.verts[rand_idx]
                ref_value = sdf[rand_idx]
                output = self.mlp(self.encoder(p))
                loss = loss_fn(output[..., 0], ref_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
        else:
            self.sdf.data = sdf.squeeze()

    @torch.no_grad()
    def reset_tet_scale(self, tet_scale=1.):
        if isinstance(tet_scale, float):
            tet_scale = torch.ones(3, device=self.device) * tet_scale
        self.tet_scale = tet_scale
        self.verts = self.verts * tet_scale

    @torch.no_grad()
    def generate_edges(self):
        # six edges for each tetrahedron.
        edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                             dtype=torch.long, device=self.device)
        all_edges = self.indices[:, edges].reshape(-1, 2)  # [M * 6, 2]
        all_edges_sorted = torch.sort(all_edges, dim=1)[0]
        self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def get_sdf_deform(self):
        if self.tet_mlp:
            # predict SDF and per-vertex deformation
            pred = self.mlp(self.encoder(self.verts))
            sdf, deform = pred[:, 0], pred[:, 1:]
            return sdf, torch.tanh(deform) / (self.grid_res)
        else:
            return self.sdf, torch.tanh(self.deform) / (self.grid_res)

    def get_verts_face(self):
        sdf, deform = self.get_sdf_deform()
        verts, faces, _, _ = self.marching_tets(
            self.verts + deform, sdf, self.indices, return_uv=False)
        return verts, faces

    # def getAABB(self):
    #     return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    # def getMesh(self, material):

    #     pred = self.mlp(self.encoder(self.verts)) # predict SDF and per-vertex deformation
    #     sdf, deform = pred[:, 0], pred[:, 1:]

    #     v_deformed = self.verts + torch.tanh(deform) / (self.grid_res)

    #     verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, sdf, self.indices)

    #     imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

    #     # Run mesh operations to generate tangent space
    #     imesh = mesh.auto_normals(imesh)
    #     imesh = mesh.compute_tangents(imesh)

    #     return imesh, sdf

    # def render(self, glctx, target, lgt, opt_material, bsdf=None):

    #     # return rendered buffers, keys: ['shaded', 'kd_grad', 'occlusion'].
    #     opt_mesh, sdf = self.getMesh(opt_material)
    #     buffers = render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'],
    #                                     msaa=True, background=None, bsdf=bsdf)
    #     buffers['mesh'] = opt_mesh
    #     buffers['sdf'] = sdf

    #     return buffers

    # def tick(self, glctx, target, lgt, opt_material, loss_fn, guidance_model, text_z, iteration):

    #     # ==============================================================================================
    #     #  Render optimizable object with identical conditions
    #     # ==============================================================================================
    #     buffers = self.render(glctx, target, lgt, opt_material)

    #     mesh = buffers['mesh']

    #     # ==============================================================================================
    #     #  Compute loss
    #     # ==============================================================================================
    #     t_iter = iteration / self.opt.iter

    #     if iteration < int(self.opt.iter * 0.2):
    #         # mode = 'normal_latent'
    #         pred_rgb = buffers['normal'][..., 0:4].permute(0, 3, 1, 2).contiguous()
    #         as_latent = True
    #     elif iteration < int(self.opt.iter * 0.6):
    #         # mode = 'normal'
    #         pred_rgb = buffers['normal'][..., 0:3].permute(0, 3, 1, 2).contiguous()
    #         as_latent = False
    #     else:
    #         # mode = 'rgb'
    #         pred_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()
    #         pred_ws = buffers['shaded'][..., 3].unsqueeze(1) # [B, 1, H, W]
    #         pred_rgb = pred_rgb * pred_ws + (1 - pred_ws) * 1 # white bg
    #         as_latent = False

    #     # torch_vis_2d(pred_rgb[0])
    #     # torch_vis_2d(pred_normal[0])
    #     # torch_vis_2d(pred_ws[0])

    #     if self.opt.directional_text:
    #         all_pos = []
    #         all_neg = []
    #         for emb in text_z[target['direction']]: # list of [2, S, -1]
    #             pos, neg = emb.chunk(2) # [1, S, -1]
    #             all_pos.append(pos)
    #             all_neg.append(neg)
    #         text_embedding = torch.cat(all_pos + all_neg, dim=0) # [2b, S, -1]
    #     else:
    #         text_embedding = text_z

    #     img_loss = guidance_model.train_step(text_embedding, pred_rgb.half(), as_latent=as_latent)

    #     # img_loss = torch.tensor(0.0, device = self.device)

    #     # below are lots of regularizations...
    #     reg_loss = torch.tensor(0.0, device = self.device)

    #     if iteration < int(self.opt.iter * 0.6):
    #         # SDF regularizer
    #         sdf_weight = self.opt.sdf_regularizer - (self.opt.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
    #         sdf_loss = sdf_reg_loss(buffers['sdf'], self.all_edges).mean() * sdf_weight # Dropoff to 0.01
    #         reg_loss = reg_loss + sdf_loss

    #         # directly regularize mesh smoothness in finetuning...
    #         if iteration > int(self.opt.iter * 0.2):
    #             lap_loss = regularizer.laplace_regularizer_const(mesh.v_pos, mesh.t_pos_idx) * self.opt.laplace_scale #* min(1.0, iteration / 500)
    #             reg_loss = reg_loss + lap_loss

    #         # normal_loss = regularizer.normal_consistency(mesh.v_pos, mesh.t_pos_idx) * self.opt.laplace_scale * min(1.0, iteration / 500)
    #         # reg_loss = reg_loss + normal_loss

    #     else:
    #         # Albedo (k_d) smoothnesss regularizer
    #         # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, (iteration - int(self.opt.iter * 0.6)) / 500)

    #         # # Visibility regularizer
    #         # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, (iteration - int(self.opt.iter * 0.6)) / 500)

    #         # # Light white balance regularizer
    #         reg_loss += lgt.regularizer() * 0.005

    #     return img_loss, reg_loss


def compute_edge_to_face_mapping(attr_idx):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((attr_idx[:, 0], attr_idx[:, 1]), dim=-1),
            torch.stack((attr_idx[:, 1], attr_idx[:, 2]), dim=-1),
            torch.stack((attr_idx[:, 2], attr_idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(
            sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(attr_idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros(
            (unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:, 0] == 0
        mask1 = order[:, 0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge


@torch.cuda.amp.autocast(enabled=False)
def normal_consistency(face_normals, t_pos_idx):

    tris_per_edge = compute_edge_to_face_mapping(t_pos_idx)

    # Fetch normals for both faces sharind an edge
    n0 = face_normals[tris_per_edge[:, 0], :]
    n1 = face_normals[tris_per_edge[:, 1], :]

    # Compute error metric based on normal difference
    term = torch.clamp(torch.sum(n0 * n1, -1, keepdim=True), min=-1.0, max=1.0)
    term = (1.0 - term)

    return torch.mean(torch.abs(term))


def laplacian_uniform(verts, faces):

    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack(
        [torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(
        adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()


@torch.cuda.amp.autocast(enabled=False)
def laplacian_smooth_loss(verts, faces):
    with torch.no_grad():
        L = laplacian_uniform(verts, faces.long())
    loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss


class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.bound = opt.bound
        self.cascade = 1 + math.ceil(math.log2(opt.bound))
        self.grid_size = 128
        self.max_level = None
        self.dmtet = opt.dmtet
        self.cuda_ray = opt.cuda_ray
        self.taichi_ray = opt.taichi_ray
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor(
            [-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        self.glctx = None

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros(
                [self.cascade, self.grid_size ** 3])  # [CAS, H * H * H]
            density_bitfield = torch.zeros(
                self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8)  # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0

        # load dmtet vertices
        if self.opt.dmtet:
            self.dmtet = DMTetGeometry(opt.tet_grid_size, opt.tet_mlp, opt).to(opt.device)
            if self.opt.h <= 2048 and self.opt.w <= 2048:
                self.glctx = dr.RasterizeCudaContext()
            else:
                self.glctx = dr.RasterizeGLContext()

        if self.taichi_ray:
            from einops import rearrange
            from taichi_modules import RayMarcherTaichi
            from taichi_modules import VolumeRendererTaichi
            from taichi_modules import RayAABBIntersector as RayAABBIntersectorTaichi
            from taichi_modules import raymarching_test as raymarching_test_taichi
            from taichi_modules import composite_test as composite_test_fw
            from taichi_modules import packbits as packbits_taichi
            self.rearrange = rearrange
            self.packbits_taichi = packbits_taichi
            self.ray_aabb_intersector = RayAABBIntersectorTaichi
            self.raymarching_test_taichi = raymarching_test_taichi
            self.composite_test_fw = composite_test_fw
            self.ray_marching = RayMarcherTaichi(
                batch_size=4096)  # TODO: hard encoded batch size
            self.volume_render = VolumeRendererTaichi(
                batch_size=4096)  # TODO: hard encoded batch size
            # density grid
            density_grid = torch.zeros(
                [self.cascade, self.grid_size ** 3])  # [CAS, H * H * H]
            density_bitfield = torch.zeros(
                self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8)  # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            
        if self.opt.density_activation == 'exp':
            self.density_activation = trunc_exp
        elif self.opt.density_activation == 'softplus':
            self.density_activation =  F.softplus
        elif self.opt.density_activation == 'relu':
            self.density_activation =  F.relu
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def normal(self, x):
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)
        return normal

    @torch.no_grad()
    def density_blob(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)

        if self.opt.density_activation == 'exp':
            g = self.opt.blob_density * \
                torch.exp(- d / (2 * self.opt.blob_radius ** 2))
        else:
            g = self.opt.blob_density * \
                (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g

    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not (self.cuda_ray or self.taichi_ray):
            return
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):
        from meshutils import decimate_mesh, clean_mesh, poisson_mesh_reconstruction
        if self.opt.dmtet:
            vertices, triangles = self.dmtet.get_verts_face()
            vertices = vertices.detach().cpu().numpy()
            triangles = triangles.detach().cpu().numpy()

        else:

            if resolution is None:
                resolution = self.grid_size

            if self.cuda_ray:
                density_thresh = min(self.mean_density, self.density_thresh) \
                    if np.greater(self.mean_density, 0) else self.density_thresh
            else:
                density_thresh = self.density_thresh

            sigmas = np.zeros(
                [resolution, resolution, resolution], dtype=np.float32)

            # query
            X = torch.linspace(-1, 1, resolution).split(S)
            Y = torch.linspace(-1, 1, resolution).split(S)
            Z = torch.linspace(-1, 1, resolution).split(S)

            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        pts = torch.cat(
                            [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                        val = self.density(pts.to(self.aabb_train.device))
                        sigmas[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(
                            zs)] = val['sigma'].reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]

            logger.info(
                f'[INFO] marching cubes thresh: {density_thresh} ({sigmas.min()} ~ {sigmas.max()})')

            vertices, triangles = mcubes.marching_cubes(sigmas, density_thresh)
            vertices = vertices / (resolution - 1.0) * 2 - 1

        # clean
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        vertices, triangles = clean_mesh(
            vertices, triangles, remesh=True, remesh_size=0.01)

        # decimation
        if decimate_target > 0 and triangles.shape[0] > decimate_target:
            vertices, triangles = decimate_mesh(
                vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices).contiguous(
        ).float().to(self.aabb_train.device)
        f = torch.from_numpy(triangles).contiguous().int().to(
            self.aabb_train.device)

        # mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        # mesh.export(os.path.join(path, f'mesh.ply'))

        def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
            # v, f: torch Tensor
            device = v.device
            v_np = v.cpu().numpy()  # [N, 3]
            f_np = f.cpu().numpy()  # [M, 3]

            logger.info(
                f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uvs
            import xatlas
            import nvdiffrast.torch as dr
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4  # for faster unwrap...
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(
                uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0
            
            if self.glctx is None:
                if h <= 2048 and w <= 2048:
                    self.glctx = dr.RasterizeCudaContext()
                else:
                    self.glctx = dr.RasterizeGLContext()

            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(
                0), ft, (h, w))  # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(
                v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

            # masked query 
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            
            feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask] # [M, 3]

                # batched inference to avoid OOM
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    results_ = self.density(xyzs[head:tail])
                    all_feats.append(results_['albedo'].float())
                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0)
            
            feats = feats.view(h, w, -1)
            mask = mask.view(h, w)

            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)

            ### NN search as an antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

            # do ssaa after the NN search, in numpy
            if ssaa > 1:
                feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(os.path.join(path, f'{name}albedo.png'), feats)

            # save obj (v, vt, f /)
            obj_file = os.path.join(path, f'{name}mesh.obj')
            mtl_file = os.path.join(path, f'{name}mesh.mtl')

            logger.info(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:
                fp.write(f'mtllib {name}mesh.mtl \n')

                logger.info(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

                logger.info(
                    f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n')

                logger.info(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl mat0 \n')
                for i in range(len(f_np)):
                    fp.write(
                        f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl mat0 \n')
                fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                fp.write(f'Tr 1.000000 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0.000000 \n')
                fp.write(f'map_Kd {name}albedo.png \n')

        _export(v, f)

    def run(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        # nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        # nears.unsqueeze_(-1)
        # fars.unsqueeze_(-1)
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='sphere', min_near=self.min_near)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            if self.training:
                light_d = safe_normalize(rays_o + torch.randn(3, device=rays_o.device)) # [N, 3]
            else:
                light_d = safe_normalize(rays_o[0:1] + torch.randn(3, device=rays_o.device)) # [N, 3]
                
        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, self.opt.num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, self.opt.num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / self.opt.num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:]) # a manual clip.

        #plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        #sigmas = density_outputs['sigma'].view(N, self.opt.num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, self.opt.num_steps, -1)

        # upsample z_vals (nerf-like)
        if self.opt.upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], self.opt.upsample_steps, det=not self.training).detach() # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(-1) # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:]) # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            #new_sigmas = new_density_outputs['sigma'].view(N, self.opt.upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, self.opt.upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1) # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1) # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1)) # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        light_d = light_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        dirs = safe_normalize(dirs)
        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio, shading=shading)
        rgbs = rgbs.view(N, -1, 3) # [N, T+t, 3]
        if normals is not None:
            normals = normals.view(N, -1, 3)

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1) # [N]
        
        # calculate depth 
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1
            
        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        weights_sum = weights_sum.reshape(*prefix)

        if self.training:
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.sum(-1).mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
        if normals is not None:
            normal_image = torch.sum(
                weights.unsqueeze(-1) * (normals + 1) / 2, dim=-2)  # [N, 3], in [0, 1]
            results['normal_image'] = normal_image
    
        results['image'] = image
        results['depth'] = depth
        results['weights'] = weights
        results['weights_sum'] = weights_sum

        return results


    def run_cuda(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, binarize=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            if self.training:
                light_d = safe_normalize(rays_o[0:1] + torch.randn(3, device=rays_o.device)) # [N, 3]
            else:
                light_d = safe_normalize(rays_o[0:1] + torch.randn(3, device=rays_o.device)) # [N, 3]

        results = {}

        if self.training:
            xyzs, dirs, ts, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb, self.opt.dt_gamma, self.opt.max_steps)
            dirs = safe_normalize(dirs)

            if light_d.shape[0] > 1:
                flatten_rays = raymarching.flatten_rays(rays, xyzs.shape[0]).long()
                light_d = light_d[flatten_rays]
            
            sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            weights, weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, ts, rays, T_thresh, binarize)
            
            # normals related regularizations
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss 
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
            if normals is not None:
                _, _, _, normal_image = raymarching.composite_rays_train(sigmas.detach(), (normals + 1) / 2, ts, rays, T_thresh, binarize)
                results['normal_image'] = normal_image
            
            # weights normalization
            results['weights'] = weights

        else:
           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < self.opt.max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, ts = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, perturb if step == 0 else False, self.opt.dt_gamma, self.opt.max_steps)
                dirs = safe_normalize(dirs)
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, ts, weights_sum, depth, image, T_thresh, binarize)

                rays_alive = rays_alive[rays_alive >= 0]
                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        
        return results

    def get_sdf_albedo_for_init(self, points=None):
        output = self.density(self.dmtet.verts if points is None else points)
        sigma, albedo = output['sigma'], output['albedo']
        return sigma - self.density_thresh, albedo

    def run_dmtet(self, rays_o, rays_d, mvp, h, w, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, **kwargs):
        # mvp: [B, 4, 4]

        device = mvp.device
        campos = rays_o[:, 0, :] # only need one ray per batch

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = safe_normalize(campos + torch.randn_like(campos)).view(-1, 1, 1, 3) # [B, 1, 1, 3]

        results = {}

        verts, faces = self.dmtet.get_verts_face()

        # get normals
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

        faces = faces.int()
        
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = safe_normalize(face_normals)
        
        vn = torch.zeros_like(verts)
        vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

        vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

        # rasterization
        verts_clip = torch.bmm(F.pad(verts, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).repeat(mvp.shape[0], 1, 1), 
                               mvp.permute(0,2,1)).float()  # [B, N, 4]
        rast, rast_db = dr.rasterize(self.glctx, verts_clip, faces, (h, w))
        
        alpha, _ = dr.interpolate(torch.ones_like(verts[:, :1]).unsqueeze(0), rast, faces) # [B, H, W, 1]
        xyzs, _ = dr.interpolate(verts.unsqueeze(0), rast, faces) # [B, H, W, 3]
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, faces)
        normal = safe_normalize(normal)

        xyzs = xyzs.view(-1, 3)
        mask = (alpha > 0).view(-1).detach()

        # do the lighting here since we have normal from mesh now.
        albedo = torch.zeros_like(xyzs, dtype=torch.float32)
        if mask.any():
            masked_albedo = self.density(xyzs[mask])['albedo']
            albedo[mask] = masked_albedo.float()
        albedo = albedo.view(-1, h, w, 3)

        if shading == 'albedo':
            color = albedo
        elif shading == 'textureless':
            lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
            color = lambertian.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif shading == 'normal':
            color = (normal + 1) / 2
        else: # 'lambertian'
            lambertian = ambient_ratio + (1 - ambient_ratio)  * (normal * light_d).sum(-1).float().clamp(min=0)
            color = albedo * lambertian.unsqueeze(-1)

        color = dr.antialias(color, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
        alpha = dr.antialias(alpha, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 1]

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1
        
        if torch.is_tensor(bg_color) and len(bg_color.shape) > 1:
            bg_color = bg_color.view(-1, h, w, 3)
        
        depth = rast[:, :, :, [2]] # [B, H, W]
        color = color + (1 - alpha) * bg_color

        results['depth'] = depth        
        results['image'] = color
        results['weights_sum'] = alpha.squeeze(-1)

        normal_image = dr.antialias((normal + 1) / 2, rast, verts_clip, faces).clamp(0, 1) # [B, H, W, 3]
        results['normal_image'] = normal_image
        
        # regularizations
        if self.training:
            if self.opt.lambda_mesh_normal > 0:
                results['loss_normal'] = normal_consistency(
                    face_normals, faces)
            if self.opt.lambda_mesh_lap > 0:
                results['loss_lap'] = laplacian_smooth_loss(verts, faces)

        return results

    def run_taichi(self, rays_o, rays_d, light_d=None, ambient_ratio=1.0, shading='albedo', bg_color=None, perturb=False, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        exp_step_factor = kwargs.get('exp_step_factor', 0.)
        MAX_SAMPLES = 1024
        NEAR_DISTANCE = 0.01
        center = torch.zeros(1, 3)
        half_size = torch.ones(1, 3)
        _, hits_t, _ = self.ray_aabb_intersector.apply(rays_o, rays_d, center, half_size, 1)
        hits_t[(hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

        # TODO: should sample different light_d for each batch... but taichi end doesn't have a flatten_ray implemented currently...
        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        results = {}

        if self.training:
            rays_a, xyzs, dirs, deltas, ts, _ = self.ray_marching(rays_o, rays_d, hits_t[:, 0], self.density_bitfield, self.cascade, self.bound, exp_step_factor, self.grid_size, MAX_SAMPLES)
            dirs = safe_normalize(dirs)
            # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())
            sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
            _, weights_sum, depth, image, weights = self.volume_render(sigmas, rgbs, deltas, ts, rays_a, kwargs.get('T_threshold', 1e-4))
            
            # normals related regularizations
            if self.opt.lambda_orient > 0 and normals is not None:
                # orientation loss 
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()
            
            if self.opt.lambda_3d_normal_smooth > 0 and normals is not None:
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                results['loss_normal_perturb'] = (normals - normals_perturb).abs().mean()
            
            if normals is not None:
                _, _, _, normal_image, _ = self.volume_render(sigmas.detach(), (normals + 1) / 2, deltas, ts, rays_a, kwargs.get('T_threshold', 1e-4))
                results['normal_image'] = normal_image
            
            # weights normalization
            results['weights'] = weights

        else:
        
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = hits_t[:, 0, 0]
            step = 0
            
            min_samples = 1 if exp_step_factor == 0 else 4

            while step < self.opt.max_steps: # hard coded max step

                # count alive rays 
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                # n_step = max(min(N // n_alive, 8), 1)
                n_step = max(min(N // n_alive, 64), min_samples)

                xyzs, dirs, deltas, ts, N_eff_samples = \
                self.raymarching_test_taichi(rays_o, rays_d, hits_t[:, 0], rays_alive,
                                    self.density_bitfield, self.cascade,
                                    self.bound, exp_step_factor,
                                    self.grid_size, MAX_SAMPLES, n_step)

                xyzs = self.rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
                dirs = self.rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
                dirs = safe_normalize(dirs)
                valid_mask = ~torch.all(dirs == 0, dim=1)
                if valid_mask.sum() == 0:
                    break

                sigmas = torch.zeros(len(xyzs), device=device)
                rgbs = torch.zeros(len(xyzs), 3, device=device)
                normals = torch.zeros(len(xyzs), 3, device=device)

                sigmas[valid_mask], _rgbs, normals = self(xyzs[valid_mask], dirs[valid_mask], light_d, ratio=ambient_ratio, shading=shading)
                rgbs[valid_mask] = _rgbs.float()
                sigmas = self.rearrange(sigmas, '(n1 n2) -> n1 n2', n2=n_step)
                rgbs = self.rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=n_step)
                if normals is not None:
                    normals = self.rearrange(normals, '(n1 n2) c -> n1 n2 c', n2=n_step)

                self.composite_test_fw(sigmas, rgbs, deltas, ts, hits_t[:,0], rays_alive,
                                    kwargs.get('T_threshold', 1e-4), N_eff_samples,
                                    weights_sum, depth, image)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

        # mix background color
        if bg_color is None:
            if self.opt.bg_radius > 0:
                # use the bg model to calculate bg_color
                bg_color = self.background(rays_d) # [N, 3]
            else:
                bg_color = 1

        image = image + self.rearrange(1 - weights_sum, 'n -> n 1') * bg_color
        image = image.view(*prefix, 3)

        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        
        return results


    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not (self.cuda_ray or self.taichi_ray):
            return 
        
        ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.aabb_train.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()
                        # assign 
                        tmp_grid[cas, indices] = sigmas
        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        if self.cuda_ray:
            self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)
        elif self.taichi_ray:
            self.packbits_taichi(self.density_grid.reshape(-1).contiguous(), density_thresh, self.density_bitfield)

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f}')


    def render(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: pred_rgb: [B, N, 3]
        B, N = rays_o.shape[:2]
        device = rays_o.device

        if self.dmtet:
            results = self.run_dmtet(rays_o, rays_d, mvp, h, w, **kwargs)
        elif self.cuda_ray:
            results = self.run_cuda(rays_o, rays_d, **kwargs)
        elif self.taichi_ray:
            results = self.run_taichi(rays_o, rays_d, **kwargs)
        else:
            if staged:
                depth = torch.empty((B, N), device=device)
                image = torch.empty((B, N, 3), device=device)
                weights_sum = torch.empty((B, N), device=device)

                for b in range(B):
                    head = 0
                    while head < N:
                        tail = min(head + max_ray_batch, N)
                        results_ = self.run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                        depth[b:b+1, head:tail] = results_['depth']
                        weights_sum[b:b+1, head:tail] = results_['weights_sum']
                        image[b:b+1, head:tail] = results_['image']
                        head += max_ray_batch
                
                results = {}
                results['depth'] = depth
                results['image'] = image
                results['weights_sum'] = weights_sum

            else:
                results = self.run(rays_o, rays_d, **kwargs)

        return results

    def init_tet_from_nerf(self, reset_scale=True):
        sdf = self.get_sdf_from_nerf(reset_scale=reset_scale)
        self.dmtet.init_tet_from_sdf(sdf)
        logger.info(f'init dmtet from NeRF Done ...')


    @torch.no_grad()
    def get_sdf_from_nerf(self, reset_scale=True):
        if self.cuda_ray:
            density_thresh = min(self.mean_density, self.density_thresh)
        else:
            density_thresh = self.density_thresh

        if reset_scale:
            # init scale
            sigma = self.density(self.dmtet.verts)[
                'sigma']  # verts covers [-1, 1] now
            mask = sigma > density_thresh
            valid_verts = self.dmtet.verts[mask]
            tet_scale = valid_verts.abs().amax(dim=0) + 1e-1
            self.dmtet.reset_tet_scale(tet_scale)
        sdf = (self.density(self.dmtet.verts)[
                'sigma'] - density_thresh).clamp(-1, 1)
        return sdf
