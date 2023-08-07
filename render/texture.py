# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr

from . import util

######################################################################################
# Smooth pooling / mip computation with linear gradient upscaling
######################################################################################

class texture2d_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, texture):
        return util.avg_pool_nhwc(texture, (2,2))

    @staticmethod
    def backward(ctx, dout):
        gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.25 / dout.shape[1], 1.0 - 0.25 / dout.shape[1], dout.shape[1]*2, device="cuda"), 
                                torch.linspace(0.0 + 0.25 / dout.shape[2], 1.0 - 0.25 / dout.shape[2], dout.shape[2]*2, device="cuda"),
                                ) # indexing='ij')
        uv = torch.stack((gx, gy), dim=-1)
        return dr.texture(dout * 0.25, uv[None, ...].contiguous(), filter_mode='linear', boundary_mode='clamp')

########################################################################################################
# Simple texture class. A texture can be either 
# - A 3D tensor (using auto mipmaps)
# - A list of 3D tensors (full custom mip hierarchy)
########################################################################################################

class Texture2D(torch.nn.Module):
     # Initializes a texture from image data.
     # Input can be constant value (1D array) or texture (3D array) or mip hierarchy (list of 3d arrays)
    def __init__(self, init, min_max=None):
        super(Texture2D, self).__init__()

        if isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')
        elif isinstance(init, list) and len(init) == 1:
            init = init[0]

        if isinstance(init, list):
            self.data = list(torch.nn.Parameter(mip.clone().detach(), requires_grad=True) for mip in init)
        elif len(init.shape) == 4:
            self.data = torch.nn.Parameter(init.clone().detach(), requires_grad=True)
        elif len(init.shape) == 3:
            self.data = torch.nn.Parameter(init[None, ...].clone().detach(), requires_grad=True)
        elif len(init.shape) == 1:
            self.data = torch.nn.Parameter(init[None, None, None, :].clone().detach(), requires_grad=True) # Convert constant to 1x1 tensor
        else:
            assert False, "Invalid texture object"

        self.min_max = min_max

    # Filtered (trilinear) sample texture at a given location
    def sample(self, texc, texc_deriv, filter_mode='linear-mipmap-linear'):
        if isinstance(self.data, list):
            out = dr.texture(self.data[0], texc, texc_deriv, mip=self.data[1:], filter_mode=filter_mode)
        else:
            if self.data.shape[1] > 1 and self.data.shape[2] > 1:
                mips = [self.data]
                while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                    mips += [texture2d_mip.apply(mips[-1])]
                out = dr.texture(mips[0], texc, texc_deriv, mip=mips[1:], filter_mode=filter_mode)
            else:
                out = dr.texture(self.data, texc, texc_deriv, filter_mode=filter_mode)
        return out

    def getRes(self):
        return self.getMips()[0].shape[1:3]

    def getChannels(self):
        return self.getMips()[0].shape[3]

    def getMips(self):
        if isinstance(self.data, list):
            return self.data
        else:
            return [self.data]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        if self.min_max is not None:
            for mip in self.getMips():
                for i in range(mip.shape[-1]):
                    mip[..., i].clamp_(min=self.min_max[0][i], max=self.min_max[1][i])

    # In-place clamp with no derivative to make sure values are in valid range after training
    def normalize_(self):
        with torch.no_grad():
            for mip in self.getMips():
                mip = util.safe_normalize(mip)

########################################################################################################
# Helper function to create a trainable texture from a regular texture. The trainable weights are 
# initialized with texture data as an initial guess
########################################################################################################

def create_trainable(init, res=None, auto_mipmaps=True, min_max=None):
    with torch.no_grad():
        if isinstance(init, Texture2D):
            assert isinstance(init.data, torch.Tensor)
            min_max = init.min_max if min_max is None else min_max
            init = init.data
        elif isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')

        # Pad to NHWC if needed
        if len(init.shape) == 1: # Extend constant to NHWC tensor
            init = init[None, None, None, :]
        elif len(init.shape) == 3:
            init = init[None, ...]

        # Scale input to desired resolution.
        if res is not None:
            init = util.scale_img_nhwc(init, res)

        # Genreate custom mipchain
        if not auto_mipmaps:
            mip_chain = [init.clone().detach().requires_grad_(True)]
            while mip_chain[-1].shape[1] > 1 or mip_chain[-1].shape[2] > 1:
                new_size = [max(mip_chain[-1].shape[1] // 2, 1), max(mip_chain[-1].shape[2] // 2, 1)]
                mip_chain += [util.scale_img_nhwc(mip_chain[-1], new_size)]
            return Texture2D(mip_chain, min_max=min_max)
        else:
            return Texture2D(init, min_max=min_max)

########################################################################################################
# Convert texture to and from SRGB
########################################################################################################

def srgb_to_rgb(texture):
    return Texture2D(list(util.srgb_to_rgb(mip) for mip in texture.getMips()))

def rgb_to_srgb(texture):
    return Texture2D(list(util.rgb_to_srgb(mip) for mip in texture.getMips()))

########################################################################################################
# Utility functions for loading / storing a texture
########################################################################################################

def _load_mip2D(fn, lambda_fn=None, channels=None):
    imgdata = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')
    if channels is not None:
        imgdata = imgdata[..., 0:channels]
    if lambda_fn is not None:
        imgdata = lambda_fn(imgdata)
    return imgdata.detach().clone()

def load_texture2D(fn, lambda_fn=None, channels=None):
    base, ext = os.path.splitext(fn)
    if os.path.exists(base + "_0" + ext):
        mips = []
        while os.path.exists(base + ("_%d" % len(mips)) + ext):
            mips += [_load_mip2D(base + ("_%d" % len(mips)) + ext, lambda_fn, channels)]
        return Texture2D(mips)
    else:
        return Texture2D(_load_mip2D(fn, lambda_fn, channels))

def _save_mip2D(fn, mip, mipidx, lambda_fn):
    if lambda_fn is not None:
        data = lambda_fn(mip).detach().cpu().numpy()
    else:
        data = mip.detach().cpu().numpy()

    if mipidx is None:
        util.save_image(fn, data)
    else:
        base, ext = os.path.splitext(fn)
        util.save_image(base + ("_%d" % mipidx) + ext, data)

def save_texture2D(fn, tex, lambda_fn=None):
    if isinstance(tex.data, list):
        for i, mip in enumerate(tex.data):
            _save_mip2D(fn, mip[0,...], i, lambda_fn)
    else:
        _save_mip2D(fn, tex.data[0,...], None, lambda_fn)
