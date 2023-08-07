import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
from tqdm import tqdm 


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

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
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
        self.apply(weight_init)
     

class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=2,
                 hidden_dim=32,
                 num_layers_bg=2,
                 hidden_dim_bg=16,
                 ):
        
        super().__init__(opt)
        self.num_layers = opt.num_layers if hasattr(opt, 'num_layers') else num_layers
        self.hidden_dim = opt.hidden_dim if hasattr(opt, 'hidden_dim') else hidden_dim
        num_layers_bg = opt.num_layers_bg if hasattr(opt, 'num_layers_bg') else num_layers_bg
        hidden_dim_bg = opt.hidden_dim_bg if hasattr(opt, 'hidden_dim_bg') else hidden_dim_bg

        self.encoder, self.in_dim = get_encoder('hashgrid_taichi', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        # self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.grid_levels_mask = 0 

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency_torch', input_dim=3, multires=4) # TODO: freq encoder can be replaced by a Taichi implementation
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def common_forward(self, x):

        # sigma
        h = self.encoder(x, bound=self.bound)
        
        # Feature masking for coarse-to-fine training
        if self.grid_levels_mask > 0:
            h_mask: torch.Tensor = torch.arange(self.in_dim, device=h.device) < self.in_dim - self.grid_levels_mask * self.level_dim  # (self.in_dim)
            h_mask = h_mask.reshape(1, self.in_dim).float()  # (1, self.in_dim)
            h = h * h_mask  # (N, self.in_dim)

        h = self.sigma_net(h)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma, albedo = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading
            normal = self.normal(x)

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
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr * self.opt.lr_scale_nerf},
            # {'params': self.normal_net.parameters(), 'lr': lr},
        ]        

        if self.opt.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet:
            params.append({'params': self.dmtet.parameters(), 'lr': lr})

        return params
