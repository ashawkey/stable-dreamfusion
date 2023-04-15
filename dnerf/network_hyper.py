import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=32,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_ambient=5,
                 hidden_dim_ambient=128,
                 ambient_dim=1,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # ambient network
        self.num_layers_ambient = num_layers_ambient
        self.hidden_dim_ambient = hidden_dim_ambient
        self.ambient_dim = ambient_dim
        self.encoder_time, self.in_dim_time = get_encoder(encoding_time, input_dim=1, multires=6)
        
        ambient_net = []
        for l in range(num_layers_ambient):
            if l == 0:
                in_dim = self.in_dim_time
            else:
                in_dim = hidden_dim_ambient
            
            if l == num_layers_ambient - 1:
                out_dim = self.ambient_dim
            else:
                out_dim = hidden_dim_ambient
            
            ambient_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.ambient_net = nn.ModuleList(ambient_net)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3+self.ambient_dim, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
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

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]

        # time --> ambient
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        # if enc_t.shape[0] == 1:
        #     enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']
        ambient = enc_t
        for l in range(self.num_layers_ambient):
            ambient = self.ambient_net[l](ambient)
            if l != self.num_layers_ambient - 1:
                ambient = F.relu(ambient, inplace=True)

        ambient = F.tanh(ambient) * self.bound
        x = torch.cat([x, ambient.repeat(x.shape[0], 1)], dim=1)
        
        # sigma
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

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

        return sigma, rgbs, None

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        results = {}

        # time --> ambient
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        ambient = enc_t
        for l in range(self.num_layers_ambient):
            ambient = self.ambient_net[l](ambient)
            if l != self.num_layers_ambient - 1:
                ambient = F.relu(ambient, inplace=True)

        ambient = F.tanh(ambient) * self.bound
        x = torch.cat([x, ambient.repeat(x.shape[0], 1)], dim=1)
        
        # sigma
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        return results

    def background(self, x, d):
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

        return rgbs

    
    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
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

        return rgbs        

    # optimizer utils
    def get_params(self, lr, lr_net):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.ambient_net.parameters(), 'lr': lr_net},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})
        
        return params
