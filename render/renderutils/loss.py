# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

#----------------------------------------------------------------------------
# HDR image losses
#----------------------------------------------------------------------------

def _tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)

def _SMAPE(img, target, eps=0.01):
    nom = torch.abs(img - target)
    denom = torch.abs(img) + torch.abs(target) + 0.01
    return torch.mean(nom / denom)

def _RELMSE(img, target, eps=0.1):
    nom = (img - target) * (img - target)
    denom = img * img + target * target + 0.1 
    return torch.mean(nom / denom)

def image_loss_fn(img, target, loss, tonemapper):
    if tonemapper == 'log_srgb':
        img    = _tonemap_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
        target = _tonemap_srgb(torch.log(torch.clamp(target, min=0, max=65535) + 1))

    if loss == 'mse':
        return torch.nn.functional.mse_loss(img, target)
    elif loss == 'smape':
        return _SMAPE(img, target)
    elif loss == 'relmse':
        return _RELMSE(img, target)
    else:
        return torch.nn.functional.l1_loss(img, target)
