import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

from encoding import get_encoder

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

class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency", # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=5, # a deeper MLP is very necessary for performance.
                 hidden_dim_deform=128
                 ):
        
        super().__init__(opt)
        print('dnerf')
        self.bound = opt.bound

        # deformation network
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding_deform, multires=10)
        self.encoder_time, self.in_dim_time = get_encoder(encoding_time, input_dim=1, multires=6)
        self.density_scale = 1
         # time stamps for density grid
        self.time_size = 64
        self.times = ((torch.arange(self.time_size, dtype=torch.float32) + 0.5) / self.time_size).view(-1, 1, 1) # [T, 1, 1]
        self.density_grid = torch.zeros(self.time_size, self.cascade, self.grid_size ** 3) # [T, CAS, H * H * H]

        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform + self.in_dim_time # grid dim + time
            else:
                in_dim = hidden_dim_deform
            
            if l == num_layers_deform - 1:
                out_dim = 3 # deformation for xyz
            else:
                out_dim = hidden_dim_deform
            
            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)


        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * self.bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.in_dim_time + self.in_dim_deform # concat everything
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True) # nn.ModuleList(bg_net)
        else:
            self.bg_net = None

        print('dnerf init done')

    def forward(self, x, d, t, l=None, ratio=1, shading='albedo'):
        print('forward 1')
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]

        # deform
        enc_ori_x = self.encoder_deform(x, bound=self.bound) # [N, C]
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        
        x = x + deform

        # sigma
        x = self.encoder(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        print('foward return')
        return sigma, rgbs, deform

    def density(self, x, t=None):
        print('density 1')
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        results = {}

        # deformation
        enc_ori_x = self.encoder_deform(x, bound=self.bound) # [N, C]
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        
        x = x + deform
        results['deform'] = deform
        
        # sigma
        x = self.encoder(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        print('density return')
        return results

    def background(self, d, x):
        print('background 1')
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        print('background return')
        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        print('color 1')
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        print('color return')
        return rgbs        

    # optimizer utils
    def get_params(self, lr, lr_net):
        print('get_params 1')

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_deform.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.deform_net.parameters(), 'lr': lr_net},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})
        
        print('get_params return')
        return params