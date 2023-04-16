import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer
import raymarching


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 resolution=[128] * 3,
                 degree=4,
                #  rank_vec_density=[64],
                #  rank_mat_density=[16],
                #  rank_vec=[64],
                #  rank_mat=[64],
                 rank_vec_density=[64, 64, 64, 64, 64],
                 rank_mat_density=[0, 4, 8, 12, 16],
                 rank_vec=[64, 64, 64, 64, 64],
                 rank_mat=[0, 4, 16, 32, 64],
                 bg_resolution=[512, 512],
                 bg_rank=8
                 ):
        super().__init__(opt)

        self.resolution = resolution

        self.degree = degree
        self.encoder_dir, self.enc_dir_dim = get_encoder('sphere_harmonics', degree=self.degree)
        self.out_dim = 3 * self.enc_dir_dim # only color dim

        # group list in list for composition
        self.rank_vec_density = [rank_vec_density]
        self.rank_mat_density = [rank_mat_density]
        self.rank_vec = [rank_vec]
        self.rank_mat = [rank_mat]

        # all components are divided into K groups
        assert len(rank_vec) == len(rank_mat) == len(rank_vec_density) == len(rank_mat_density)

        self.K = [len(rank_vec)]

        # utility
        self.group_vec_density = [np.diff(rank_vec_density, prepend=0)]
        self.group_mat_density = [np.diff(rank_mat_density, prepend=0)]
        self.group_vec = [np.diff(rank_vec, prepend=0)]
        self.group_mat = [np.diff(rank_mat, prepend=0)]

        self.mat_ids = [[0, 1], [0, 2], [1, 2]]
        self.vec_ids = [2, 1, 0]

        # allocate params

        self.U_vec_density = nn.ParameterList() 
        self.S_vec_density = nn.ParameterList()

        for k in range(self.K[0]):
            if self.group_vec_density[0][k] > 0:
                for i in range(3):                
                    vec_id = self.vec_ids[i]
                    w = torch.randn(self.group_vec_density[0][k], self.resolution[vec_id]) * 0.2 # [R, H]
                    self.U_vec_density.append(nn.Parameter(w.view(1, self.group_vec_density[0][k], self.resolution[vec_id], 1))) # [1, R, H, 1]
                w = torch.ones(1, self.group_vec_density[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_vec_density.append(nn.Parameter(w))

        self.U_mat_density = nn.ParameterList() 
        self.S_mat_density = nn.ParameterList()

        
        for k in range(self.K[0]):
            if self.group_mat_density[0][k] > 0:
                for i in range(3):
                    mat_id_0, mat_id_1 = self.mat_ids[i]
                    w = torch.randn(self.group_mat_density[0][k], self.resolution[mat_id_1] * self.resolution[mat_id_0]) * 0.2 # [R, HW]
                    self.U_mat_density.append(nn.Parameter(w.view(1, self.group_mat_density[0][k], self.resolution[mat_id_1], self.resolution[mat_id_0]))) # [1, R, H, W]
                w = torch.ones(1, self.group_mat_density[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_mat_density.append(nn.Parameter(w))

        self.U_vec = nn.ParameterList() 
        self.S_vec = nn.ParameterList()

        for k in range(self.K[0]):
            if self.group_vec[0][k] > 0:
                for i in range(3):                
                    vec_id = self.vec_ids[i]
                    w = torch.randn(self.group_vec[0][k], self.resolution[vec_id]) * 0.2 # [R, H]
                    self.U_vec.append(nn.Parameter(w.view(1, self.group_vec[0][k], self.resolution[vec_id], 1))) # [1, R, H, 1]
                w = torch.ones(self.out_dim, self.group_vec[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_vec.append(nn.Parameter(w))

        self.U_mat = nn.ParameterList() 
        self.S_mat = nn.ParameterList()

        for k in range(self.K[0]):
            if self.group_mat[0][k] > 0:
                for i in range(3):
                    mat_id_0, mat_id_1 = self.mat_ids[i]
                    w = torch.randn(self.group_mat[0][k], self.resolution[mat_id_1] * self.resolution[mat_id_0]) * 0.2 # [R, HW]
                    self.U_mat.append(nn.Parameter(w.view(1, self.group_mat[0][k], self.resolution[mat_id_1], self.resolution[mat_id_0]))) # [1, R, H, W]
                w = torch.ones(self.out_dim, self.group_mat[0][k])
                torch.nn.init.kaiming_normal_(w)
                self.S_mat.append(nn.Parameter(w))

        # flag
        self.finalized = False if self.K[0] != 1 else True

        # background model
        if self.bg_radius > 0:
            
            self.bg_resolution = bg_resolution
            self.bg_rank = bg_rank
            self.bg_mat = nn.Parameter(0.2 * torch.randn((1, bg_rank, bg_resolution[0], bg_resolution[1]))) # [1, R, H, W]

            w = torch.ones(self.out_dim, bg_rank) # just color
            torch.nn.init.kaiming_normal_(w)
            self.bg_S = nn.Parameter(w)


    def compute_features_density(self, x, KIN=-1, residual=False, oid=0):
        # x: [N, 3], in [-1, 1]
        # return: [K, N, out_dim]

        prefix = x.shape[:-1]
        N = np.prod(prefix)

        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]

        # calculate first K blocks
        #if KIN <= 0:
        KIN = self.K[oid]
            
        # loop all blocks 
        if residual:
            outputs = []

        last_y = None

        offset_vec = oid
        offset_mat = oid

        for k in range(KIN):

            y = 0

            if self.group_vec_density[oid][k]:
                vec_feat = F.grid_sample(self.U_vec_density[3 * offset_vec + 0], vec_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec_density[3 * offset_vec + 1], vec_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec_density[3 * offset_vec + 2], vec_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (self.S_vec_density[offset_vec] @ vec_feat)

                offset_vec += 1

            if self.group_mat_density[oid][k]:
                mat_feat = F.grid_sample(self.U_mat_density[3 * offset_mat + 0], mat_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat_density[3 * offset_mat + 1], mat_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat_density[3 * offset_mat + 2], mat_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (self.S_mat_density[offset_mat] @ mat_feat) # [out_dim, N]

                offset_mat += 1

            if last_y is not None:
                y = y + last_y

            if residual:
                outputs.append(y)

            last_y = y
        
        if residual:
            outputs = torch.stack(outputs, dim=0).permute(0, 2, 1).contiguous().view(KIN, *prefix, -1) # [K, out_dim, N] --> [K, N, out_dim]
        else:
            outputs = last_y.permute(1, 0).contiguous().view(*prefix, -1) # [out_dim, N] --> [N, out_dim]
        
        return outputs

    def compute_features(self, x, KIN=-1, residual=False, oid=0):
        # x: [N, 3], in [-1, 1]
        # return: [K, N, out_dim]

        prefix = x.shape[:-1]
        N = np.prod(prefix)

        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).view(3, -1, 1, 2)

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]

        # calculate first K blocks
        #if KIN <= 0:
        KIN = self.K[oid]
            
        # loop all blocks 
        if residual:
            outputs = []

        last_y = None

        offset_vec = oid
        offset_mat = oid

        for k in range(KIN):

            y = 0

            if self.group_vec[oid][k]:
                vec_feat = F.grid_sample(self.U_vec[3 * offset_vec + 0], vec_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec[3 * offset_vec + 1], vec_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_vec[3 * offset_vec + 2], vec_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (self.S_vec[offset_vec] @ vec_feat)

                offset_vec += 1

            if self.group_mat[oid][k]:
                mat_feat = F.grid_sample(self.U_mat[3 * offset_mat + 0], mat_coord[[0]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat[3 * offset_mat + 1], mat_coord[[1]], align_corners=False).view(-1, N) * \
                           F.grid_sample(self.U_mat[3 * offset_mat + 2], mat_coord[[2]], align_corners=False).view(-1, N) # [r, N]

                y = y + (self.S_mat[offset_mat] @ mat_feat) # [out_dim, N]

                offset_mat += 1

            if last_y is not None:
                y = y + last_y

            if residual:
                outputs.append(y)

            last_y = y
        
        if residual:
            outputs = torch.stack(outputs, dim=0).permute(0, 2, 1).contiguous().view(KIN, *prefix, -1) # [K, out_dim, N] --> [K, N, out_dim]
        else:
            outputs = last_y.permute(1, 0).contiguous().view(*prefix, -1) # [out_dim, N] --> [N, out_dim]
        
        return outputs


    def normalize_coord(self, x, oid=0):
        
        if oid == 0:
            aabb = self.aabb_train
        else:
            tr = getattr(self, f'T_{oid}') # [4, 4] transformation matrix
            x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1) # to homo
            x = (x @ tr.T)[:, :3] # [N, 4] --> [N, 3]

            aabb = getattr(self, f'aabb_{oid}')

        return 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1] in bbox
            

    def normalize_dir(self, d, oid=0):
        if oid != 0:
            tr = getattr(self, f'R_{oid}') # [3, 3] rotation matrix
            d = d @ tr.T
        return d

    
    def forward(self, x, d, KIN=-1, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        N = x.shape[0]

        # single object
        if len(self.K) == 1:

            x_model = self.normalize_coord(x)
            feats_density = self.compute_features_density(x_model, KIN, residual=self.training) # [K, N, 1]
            sigma = trunc_exp(feats_density).squeeze(-1) # [K, N]

            enc_d = self.encoder_dir(d) # [N, C]

            h = self.compute_features(x_model, KIN, residual=self.training) # [K, N, 3C]
            h = h.view(N, 3, self.degree ** 2) # [K, N, 3, C]
            h = (h * enc_d.unsqueeze(1)).sum(-1) # [K, N, 3]

            rgb = torch.sigmoid(h) # [K, N, 3] 

            return sigma, rgb, None

        # multi-object (composed scene), do not support rank-residual training for now.
        else:
            
            sigma_list = []
            h_list = []

            sigma_all = 0
            rgb_all = 0


            for oid in range(1, len(self.K)):
                x_model = self.normalize_coord(x, oid=oid)

                feats_density = self.compute_features_density(x_model, -1, residual=False, oid=oid) # [N, 1]

                sigma = trunc_exp(feats_density).squeeze(-1) # [N]
                sigma_list.append(sigma.detach().clone())

                sigma_all += sigma

                d_model = self.normalize_dir(d, oid=oid)
                enc_d = self.encoder_dir(d_model) # [N, C]

                h = self.compute_features(x_model, -1, residual=False, oid=oid) # [N, 3C]
                h = h.view(N, 3, self.degree ** 2)
                h = (h * enc_d.unsqueeze(1)).sum(-1) # [N, 3]

                h_list.append(h)


            ws = torch.stack(sigma_list, dim=0) # [O, N]
            ws = F.softmax(ws, dim=0)

            for oid in range(1, len(self.K)):
                rgb_all += h_list[oid - 1] * ws[oid - 1].unsqueeze(-1)

            rgb_all = torch.sigmoid(rgb_all)

            return sigma_all, rgb_all, None


    def density(self, x, KIN=-1):
        # x: [N, 3], in [-bound, bound]

        if len(self.K) == 1:
        
            x_model = self.normalize_coord(x)
            feats_density = self.compute_features_density(x_model, KIN, residual=False) # [N, 1 + 3C]
            sigma = trunc_exp(feats_density).squeeze(-1) # [N]

            return {
                'sigma': sigma,
            }

        else:

            sigma_all = 0
            for oid in range(1, len(self.K)):
                x_model = self.normalize_coord(x, oid=oid)
                feats_density = self.compute_features_density(x_model, -1, residual=False, oid=oid) # [N, 1]
                sigma = trunc_exp(feats_density).squeeze(-1) # [N]
                sigma_all += sigma

            return {
                'sigma': sigma_all,
            }


    def background(self, x, d):
        # x: [N, 2] in [-1, 1]

        N = x.shape[0]

        h = F.grid_sample(self.bg_mat, x.view(1, N, 1, 2), align_corners=False).view(-1, N) # [R, N]
        h = (self.bg_S @ h).T.contiguous() # [3C, N] --> [N, 3C]
        enc_d = self.encoder_dir(d)

        h = h.view(N, 3, -1)
        h = (h * enc_d.unsqueeze(1)).sum(-1) # [N, 3]
        
        # sigmoid activation for rgb
        rgb = torch.sigmoid(h)

        return rgb


    # L1 penalty for loss
    def density_loss(self):
        loss = 0
        for i in range(len(self.U_vec_density)):
            loss = loss + torch.mean(torch.abs(self.U_vec_density[i]))
        for i in range(len(self.U_mat_density)):
            loss = loss + torch.mean(torch.abs(self.U_mat_density[i]))
        return loss
    

    # upsample utils
    @torch.no_grad()
    def upsample_model(self, resolution):

        for i in range(len(self.U_vec_density)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec_density[i] = nn.Parameter(F.interpolate(self.U_vec_density[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=False))

        for i in range(len(self.U_mat_density)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat_density[i] = nn.Parameter(F.interpolate(self.U_mat_density[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=False))

        for i in range(len(self.U_vec)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec[i] = nn.Parameter(F.interpolate(self.U_vec[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=False))

        for i in range(len(self.U_mat)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat[i] = nn.Parameter(F.interpolate(self.U_mat[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=False))

        self.resolution = resolution

        print(f'[INFO] upsampled to {resolution}')

    @torch.no_grad()
    def shrink_model(self):
        # shrink aabb_train and the model so it only represents the space inside aabb_train.

        half_grid_size = self.bound / self.grid_size
        thresh = min(self.density_thresh, self.mean_density)

        # get new aabb from the coarsest density grid (TODO: from the finest that covers current aabb?)
        valid_grid = self.density_grid[self.cascade - 1] > thresh # [N]
        valid_pos = raymarching.morton3D_invert(torch.nonzero(valid_grid)) # [Nz] --> [Nz, 3], in [0, H - 1]
        #plot_pointcloud(valid_pos.detach().cpu().numpy())
        valid_pos = (2 * valid_pos / (self.grid_size - 1) - 1) * (self.bound - half_grid_size) # [Nz, 3], in [-b+hgs, b-hgs]
        min_pos = valid_pos.amin(0) - half_grid_size # [3]
        max_pos = valid_pos.amax(0) + half_grid_size # [3]

        # shrink model
        reso = torch.LongTensor(self.resolution).to(self.aabb_train.device)
        units = (self.aabb_train[3:] - self.aabb_train[:3]) / reso
        tl = (min_pos - self.aabb_train[:3]) / units
        br = (max_pos - self.aabb_train[:3]) / units
        tl = torch.round(tl).long().clamp(min=0)
        br = torch.minimum(torch.round(br).long(), reso)
        
        for i in range(len(self.U_vec_density)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec_density[i] = nn.Parameter(self.U_vec_density[i].data[..., tl[vec_id]:br[vec_id], :])
        
        for i in range(len(self.U_mat_density)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat_density[i] = nn.Parameter(self.U_mat_density[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        for i in range(len(self.U_vec)):
            vec_id = self.vec_ids[i % 3]
            self.U_vec[i] = nn.Parameter(self.U_vec[i].data[..., tl[vec_id]:br[vec_id], :])
        
        for i in range(len(self.U_mat)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.U_mat[i] = nn.Parameter(self.U_mat[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        self.aabb_train = torch.cat([min_pos, max_pos], dim=0) # [6]

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {self.aabb_train.cpu().numpy().tolist()}')

    
    @torch.no_grad()
    def finalize_group(self, U, S):

        if len(U) == 0 or len(S) == 0:
            return nn.ParameterList(), nn.ParameterList()

        # sort rank inside each group
        for i in range(len(S)):
            importance = S[i].abs().sum(0) # [C, R] --> [R]
            for j in range(3):
                importance *= U[3 * i + j].view(importance.shape[0], -1).norm(dim=-1) # [R, H] --> [R]   
        
            inds = torch.argsort(importance, descending=True) # important first

            S[i] = nn.Parameter(S[i].data[:, inds])
            for j in range(3):
                U[3 * i + j] = nn.Parameter(U[3 * i + j].data[:, inds])

        # fuse rank across all groups

        S = nn.ParameterList([
            nn.Parameter(torch.cat([s.data for s in S], dim=1))
        ])

        U = nn.ParameterList([
            nn.Parameter(torch.cat([v.data for v in U[0::3]], dim=1)),
            nn.Parameter(torch.cat([v.data for v in U[1::3]], dim=1)),
            nn.Parameter(torch.cat([v.data for v in U[2::3]], dim=1)),
        ])

        return U, S


    # finalize model parameters (fuse all groups) for faster inference, but no longer allow rank-residual training.
    @torch.no_grad()
    def finalize(self):
        self.U_vec_density, self.S_vec_density = self.finalize_group(self.U_vec_density, self.S_vec_density)
        self.U_mat_density, self.S_mat_density = self.finalize_group(self.U_mat_density, self.S_mat_density)
        self.U_vec, self.S_vec = self.finalize_group(self.U_vec, self.S_vec)
        self.U_mat, self.S_mat = self.finalize_group(self.U_mat, self.S_mat)

        # update states        
        self.rank_vec_density[0] = [self.rank_vec_density[0][-1]]
        self.rank_mat_density[0] = [self.rank_mat_density[0][-1]]
        self.rank_vec[0] = [self.rank_vec[0][-1]]
        self.rank_mat[0] = [self.rank_mat[0][-1]]

        self.group_vec_density[0] = self.rank_vec_density[0]
        self.group_mat_density[0] = self.rank_mat_density[0]
        self.group_vec[0] = self.rank_vec[0]
        self.group_mat[0] = self.rank_mat[0]

        self.K[0] = 1

        self.finalized = True

    
    # assume finalized (sorted), simply slicing!
    @torch.no_grad()
    def compress_group(self, U, S, rank):
        if rank == 0:
            return nn.ParameterList(), nn.ParameterList()
        S[0] = nn.Parameter(S[0].data[:, :rank].clone()) # clone is necessary, slicing won't change storage!
        for i in range(3):
            U[i] = nn.Parameter(U[i].data[:, :rank].clone())
        return U, S

    @torch.no_grad()
    def compress(self, ranks):
        # ranks: (density_vec, density_mat, color_vec, color_mat)
        if not self.finalized:
            self.finalize()
        
        self.U_vec_density, self.S_vec_density = self.compress_group(self.U_vec_density, self.S_vec_density, ranks[0])
        self.U_mat_density, self.S_mat_density = self.compress_group(self.U_mat_density, self.S_mat_density, ranks[1])
        self.U_vec, self.S_vec = self.compress_group(self.U_vec, self.S_vec, ranks[2])
        self.U_mat, self.S_mat = self.compress_group(self.U_mat, self.S_mat, ranks[3])

        # update states
        self.rank_vec_density[0] = [ranks[0]]
        self.rank_mat_density[0] = [ranks[1]]
        self.rank_vec[0] = [ranks[2]]
        self.rank_mat[0] = [ranks[3]]

        self.group_vec_density[0] = self.rank_vec_density[0]
        self.group_mat_density[0] = self.rank_mat_density[0]
        self.group_vec[0] = self.rank_vec[0]
        self.group_mat[0] = self.rank_mat[0]

    @torch.no_grad()
    def compose(self, other, R=None, s=None, t=None): 
        if not self.finalized:
            self.finalize()
        if not other.finalized:
            other.finalize()

        # parameters
        self.U_vec_density.extend(other.U_vec_density)
        self.S_vec_density.extend(other.S_vec_density)

        self.U_mat_density.extend(other.U_mat_density)
        self.S_mat_density.extend(other.S_mat_density)

        self.U_vec.extend(other.U_vec)
        self.S_vec.extend(other.S_vec)

        self.U_mat.extend(other.U_mat)
        self.S_mat.extend(other.S_mat)

        # states
        self.rank_vec_density.extend(other.rank_vec_density)
        self.rank_mat_density.extend(other.rank_mat_density)
        self.rank_vec.extend(other.rank_vec)
        self.rank_mat.extend(other.rank_mat)

        self.group_vec_density.extend(other.group_vec_density)
        self.group_mat_density.extend(other.group_mat_density)
        self.group_vec.extend(other.group_vec)
        self.group_mat.extend(other.group_mat)

        self.K.extend(other.K)

        # transforms
        oid = len(self.K) - 1

        # R: a [3, 3] rotation matrix in SO(3)
        if R is None:
            R = torch.eye(3, dtype=torch.float32)
        elif isinstance(R, np.ndarray):
            R = torch.from_numpy(R.astype(np.float32))
        else: # tensor
            R = R.float()

        # s is a scalar scaling factor
        if s is None:
            s = 1
        
        # t is a [3] translation vector
        if t is None:
            t = torch.zeros(3, dtype=torch.float32)
        elif isinstance(t, np.ndarray):
            t = torch.from_numpy(t.astype(np.float32))
        else: # tensor
            t = t.float()

        # T: the [4, 4] transformation matrix
        # first scale & rotate, then translate.
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R * s
        T[:3, 3] = t
        
        # T is the model matrix, but we want the matrix to transform rays, i.e., the inversion.
        T = torch.inverse(T).to(self.aabb_train.device)
        R = R.T.to(self.aabb_train.device)
        
        self.register_buffer(f'T_{oid}', T)
        self.register_buffer(f'R_{oid}', R)
        self.register_buffer(f'aabb_{oid}', other.aabb_train)
        
        # update density grid multiple times to make sure it is accurate
        # TODO: 3 is very empirical...
        for _ in range(3):
            self.update_extra_state()
        

    # optimizer utils
    def get_params(self, lr1, lr2):
        params = [
            {'params': self.U_vec_density, 'lr': lr1},
            {'params': self.S_vec_density, 'lr': lr2},
            {'params': self.U_mat_density, 'lr': lr1}, 
            {'params': self.S_mat_density, 'lr': lr2},
            {'params': self.U_vec, 'lr': lr1},
            {'params': self.S_vec, 'lr': lr2},
            {'params': self.U_mat, 'lr': lr1}, 
            {'params': self.S_mat, 'lr': lr2},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.bg_mat, 'lr': lr1})
            params.append({'params': self.bg_S, 'lr': lr2})
        return params
        