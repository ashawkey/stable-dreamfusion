# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import numpy as np
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from . import light

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos, gb_mask,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        lgt,
        material,
        bsdf
    ):

    ################################################################################
    # Texture lookups
    ################################################################################
    perturbed_nrm = None
    if 'kd_ks_normal' in material:
        # Combined texture, used for MLPs because lookups are expensive
        all_tex_jitter = material['kd_ks_normal'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
        all_tex = material['kd_ks_normal'].sample(gb_pos)
        assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
        kd, ks, perturbed_nrm = all_tex[..., :-6], all_tex[..., -6:-3], all_tex[..., -3:]
        # Compute albedo (kd) gradient, used for material regularizer
        kd_grad    = torch.sum(torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]), dim=-1, keepdim=True) / 3
    else:
        kd_jitter  = material['kd'].sample(gb_texc + torch.normal(mean=0, std=0.005, size=gb_texc.shape, device="cuda"), gb_texc_deriv)
        kd = material['kd'].sample(gb_texc, gb_texc_deriv)
        ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
        if 'normal' in material:
            perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)
        kd_grad    = torch.sum(torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True) / 3

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    kd = kd[..., 0:3]

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    if 'no_perturbed_nrm' in material and material['no_perturbed_nrm']:
        perturbed_nrm = None

    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert 'bsdf' in material or bsdf is not None, "Material must specify a BSDF type"
    bsdf = material['bsdf'] if bsdf is None else bsdf
    if bsdf == 'pbr':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'diffuse':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=False)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'normal':
        shaded_col = (gb_normal + 1.0)*0.5
    elif bsdf == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    elif bsdf == 'kd':
        shaded_col = kd
    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf
    
    # Return multiple buffers
    buffers = {
        'shaded'    : torch.cat((shaded_col, alpha), dim=-1),
        'kd_grad'   : torch.cat((kd_grad, alpha), dim=-1),
        'occlusion' : torch.cat((ks[..., :1], alpha), dim=-1),
        'normal'    : torch.cat(((gb_normal + 1.0) * 0.5, gb_mask), dim=-1),
    }
    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())
    gb_mask, _ = interpolate(torch.ones_like(mesh.v_pos[None, ..., :1]), rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    ################################################################################
    # Shade
    ################################################################################

    buffers = shade(gb_pos, gb_mask, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, view_pos, lgt, mesh.material, bsdf)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================

def render_mesh(
        ctx,
        mesh,
        mtx_in, # mvp, [B, 4, 4]
        view_pos, # cam pos, [B, 3]
        lgt,
        resolution, # [2] (check the custom collate in dataset/dataset.py)
        spp         = 1,
        num_layers  = 1, # always 1
        msaa        = False,
        background  = None, 
        bsdf        = None
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos) # [B, 1, 1, 3]

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in) # just the mvp transform, [1, N, 3]

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, lgt, resolution, spp, msaa, bsdf), rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        elif key == 'normal':
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"

    mask = (rast[..., -1:] > 0).float()
    kd = all_tex[..., :-6]
    ks = all_tex[..., -6:-3]    
    perturbed_nrm = util.safe_normalize(all_tex[..., -3:])

    # antialiasing
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    mask_np = mask.cpu().numpy() > 0

    inpaint_region = binary_dilation(mask_np, iterations=3)
    inpaint_region[mask_np] = 0

    search_region = mask_np.copy()
    not_search_region = binary_erosion(search_region, iterations=2)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    inpaint_coords = torch.from_numpy(inpaint_coords).long().to(kd.device)
    target_coords = torch.from_numpy(search_coords[indices[:, 0]]).long().to(kd.device)

    kd[tuple(inpaint_coords.T)] = kd[tuple(target_coords.T)]
    ks[tuple(inpaint_coords.T)] = ks[tuple(target_coords.T)]
    perturbed_nrm[tuple(inpaint_coords.T)] = perturbed_nrm[tuple(target_coords.T)]

    return mask, kd, ks, perturbed_nrm
