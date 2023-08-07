import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer, MLP

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
from tqdm import tqdm 
import logging


logger = logging.getLogger(__name__)


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 level_dim=2
                 ):
        
        super().__init__(opt)

        self.num_layers = opt.num_layers if hasattr(opt, 'num_layers') else num_layers
        self.hidden_dim = opt.hidden_dim if hasattr(opt, 'hidden_dim') else hidden_dim
        self.level_dim = opt.level_dim if hasattr(opt, 'level_dim') else level_dim
        num_layers_bg = opt.num_layers_bg if hasattr(opt, 'num_layers_bg') else num_layers_bg
        hidden_dim_bg = opt.hidden_dim_bg if hasattr(opt, 'hidden_dim_bg') else hidden_dim_bg

        if self.opt.grid_type == 'hashgrid':
            self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')
        elif self.opt.grid_type == 'tilegrid':
            self.encoder, self.in_dim = get_encoder(
                'tiledgrid', 
                input_dim=3,
                level_dim=self.level_dim,
                log2_hashmap_size=16,
                num_levels=16,
                desired_resolution= 2048 * self.bound,
            )
        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        # self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        # masking
        self.grid_levels_mask = 0 

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def common_forward(self, x):

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
            if shading == 'normal':
                color = (normal + 1) / 2
            else:
                lambertian = ratio + (1 - ratio) * (normal * l).sum(-1).clamp(min=0) # [N,]
                if shading == 'textureless':
                    color = lambertian.unsqueeze(-1).repeat(1, 3)
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

    def reset_sigmanet(self):
        self.sigma_net.reset_parameters()

    def init_nerf_from_sdf_color(self, rpst, albedo, 
                                 points=None, pretrain_iters=10000, lr=0.001, rpst_type='sdf', 
                                 ):
        self.reset_sigmanet()
        # matching optimization
        self.train()
        self.grid_levels_mask = 0
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)

        milestones = [int(0.4 * pretrain_iters), int(0.8 * pretrain_iters)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        rpst = rpst.squeeze().clamp(0, 1)

        # rpst = torch.ones_like(rpst) * 0.4       
        pbar = tqdm(range(pretrain_iters), desc="NeRF sigma optimization")
        rgb_loss = torch.tensor(0, device=rpst.device) 
        for i in pbar:
            output = self.density(points)
            if rpst_type == 'sdf':
                pred_rpst = output['sigma'] - self.density_thresh
            else:
                pred_rpst = output['sigma']
            sdf_loss = loss_fn(pred_rpst, rpst)
            
            if albedo is not None: 
                pred_albedo = output['albedo']
                rgb_loss = loss_fn(pred_albedo, albedo)
                loss = 10 * sdf_loss + rgb_loss
            else:
                loss = sdf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=loss.item(), rgb_loss=rgb_loss.item(), sdf_loss=sdf_loss.item())
        logger.info(f'lr: {lr} Accuracy: (pred_rpst[rpst>0]>0).sum() / (rpst>0).sum()')
        pbar = tqdm(range(pretrain_iters), desc="NeRF color optimization")


    def init_tet_from_sdf_color(self, sdf, colors=None, pretrain_iters=5000, lr=0.01):
        self.train()
        self.grid_levels_mask = 0
        
        self.dmtet.reset_tet(reset_scale=False) 
        self.dmtet.init_tet_from_sdf(sdf, pretrain_iters=pretrain_iters, lr=lr)

        if colors is not None:
            self.reset_sigmanet()
            loss_fn = torch.nn.MSELoss()
            pretrain_iters = 5000
            optimizer = torch.optim.Adam(list(self.parameters()), lr=0.01)
            pbar =  tqdm(range(pretrain_iters), desc="NeRF color optimization")
            for i in pbar:
                pred_albedo = self.density(self.dmtet.verts)['albedo'] 
                loss = loss_fn(pred_albedo, colors)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item())
