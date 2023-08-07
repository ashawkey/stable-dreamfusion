import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
from tqdm import tqdm 


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU(inplace=True)

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out    

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.ModuleList(net)

    def reset_parameters(self):
        @torch.no_grad()
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
        self.apply(weight_init)
     
    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=5, # 5 in paper
                 hidden_dim=64, # 128 in paper
                 num_layers_bg=2, # 3 in paper
                 hidden_dim_bg=32, # 64 in paper
                 encoding='frequency_torch', # pure pytorch
                 output_dim=4, # 7 for DMTet (sdf 1 + color 3 + deform 3), 4 for NeRF
                 ):
        
        super().__init__(opt)
        self.num_layers = opt.num_layers if hasattr(opt, 'num_layers') else num_layers
        self.hidden_dim = opt.hidden_dim if hasattr(opt, 'hidden_dim') else hidden_dim
        num_layers_bg = opt.num_layers_bg if hasattr(opt, 'num_layers_bg') else num_layers_bg
        hidden_dim_bg = opt.hidden_dim_bg if hasattr(opt, 'hidden_dim_bg') else hidden_dim_bg

        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, multires=6)
        self.sigma_net = MLP(self.in_dim, output_dim, hidden_dim, num_layers, bias=True, block=ResBlock)

        self.grid_levels_mask = 0 

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = self.encoder(x, bound=self.bound, max_level=self.max_level)

        # Feature masking for coarse-to-fine training
        if self.grid_levels_mask > 0:
            h_mask: torch.Tensor = torch.arange(self.in_dim, device=h.device) < self.in_dim - self.grid_levels_mask * self.level_dim  # (self.in_dim)
            h_mask = h_mask.reshape(1, self.in_dim).float()  # (1, self.in_dim)
            h = h * h_mask  # (N, self.in_dim)

        h = self.sigma_net(h)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo

    def normal(self, x):
    
        with torch.enable_grad():
            x.requires_grad_(True)
            sigma, albedo = self.common_forward(x)
            # query gradient
            normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
        
        # normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        # normal = torch.nan_to_num(normal)

        return normal
        
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x)
            normal = None
        
        else:
            # query normal

            # sigma, albedo = self.common_forward(x)
            # normal = self.normal(x)
        
            with torch.enable_grad():
                x.requires_grad_(True)
                sigma, albedo = self.common_forward(x)
                # query gradient
                normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
            normal = safe_normalize(normal)
            # normal = torch.nan_to_num(normal)
            # normal = normal.detach()

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            # {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr * self.opt.lr_scale_nerf},
        ]        

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet:
            params.append({'params': self.dmtet.parameters(), 'lr': lr})

        return params