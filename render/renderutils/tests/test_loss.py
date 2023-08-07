# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import renderutils as ru

RES = 8
DTYPE = torch.float32

def tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)

def l1(output, target):
    x = torch.clamp(output, min=0, max=65535)
    r = torch.clamp(target, min=0, max=65535)
    x = tonemap_srgb(torch.log(x + 1))
    r = tonemap_srgb(torch.log(r + 1))
    return torch.nn.functional.l1_loss(x,r)

def relative_loss(name, ref, cuda):
	ref = ref.float()
	cuda = cuda.float()
	print(name, torch.max(torch.abs(ref - cuda) / torch.abs(ref + 1e-7)).item())

def test_loss(loss, tonemapper):
	img_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	img_ref = img_cuda.clone().detach().requires_grad_(True)
	target_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	target_ref = target_cuda.clone().detach().requires_grad_(True)

	ref_loss = ru.image_loss(img_ref, target_ref, loss=loss, tonemapper=tonemapper, use_python=True)
	ref_loss.backward()

	cuda_loss = ru.image_loss(img_cuda, target_cuda, loss=loss, tonemapper=tonemapper)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Loss: %s, %s" % (loss, tonemapper))
	print("-------------------------------------------------------------")

	relative_loss("res:", ref_loss, cuda_loss)
	relative_loss("img:", img_ref.grad, img_cuda.grad)
	relative_loss("target:", target_ref.grad, target_cuda.grad)


test_loss('l1', 'none')
test_loss('l1', 'log_srgb')
test_loss('mse', 'log_srgb')
test_loss('smape', 'none')
test_loss('relmse', 'none')
test_loss('mse', 'none')