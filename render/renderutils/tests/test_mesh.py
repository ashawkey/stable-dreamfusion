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

BATCH = 8
RES = 1024
DTYPE = torch.float32

torch.manual_seed(0)

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
	print(name, torch.max(torch.abs(ref - cuda) / torch.abs(ref)).item())

def test_xfm_points():
	points_cuda = torch.rand(1, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	points_ref = points_cuda.clone().detach().requires_grad_(True)
	mtx_cuda = torch.rand(BATCH, 4, 4, dtype=DTYPE, device='cuda', requires_grad=False)
	mtx_ref = mtx_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(BATCH, RES, 4, dtype=DTYPE, device='cuda', requires_grad=True)

	ref_out = ru.xfm_points(points_ref, mtx_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref_out, target)
	ref_loss.backward()

	cuda_out = ru.xfm_points(points_cuda, mtx_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda_out, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")

	relative_loss("res:", ref_out, cuda_out)
	relative_loss("points:", points_ref.grad, points_cuda.grad)

def test_xfm_vectors():
	points_cuda = torch.rand(1, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	points_ref = points_cuda.clone().detach().requires_grad_(True)
	points_cuda_p = points_cuda.clone().detach().requires_grad_(True)
	points_ref_p = points_cuda.clone().detach().requires_grad_(True)
	mtx_cuda = torch.rand(BATCH, 4, 4, dtype=DTYPE, device='cuda', requires_grad=False)
	mtx_ref = mtx_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(BATCH, RES, 4, dtype=DTYPE, device='cuda', requires_grad=True)

	ref_out = ru.xfm_vectors(points_ref.contiguous(), mtx_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref_out, target[..., 0:3])
	ref_loss.backward()

	cuda_out = ru.xfm_vectors(points_cuda.contiguous(), mtx_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda_out, target[..., 0:3])
	cuda_loss.backward()

	ref_out_p = ru.xfm_points(points_ref_p.contiguous(), mtx_ref, use_python=True)
	ref_loss_p = torch.nn.MSELoss()(ref_out_p, target)
	ref_loss_p.backward()
	
	cuda_out_p = ru.xfm_points(points_cuda_p.contiguous(), mtx_cuda)
	cuda_loss_p = torch.nn.MSELoss()(cuda_out_p, target)
	cuda_loss_p.backward()

	print("-------------------------------------------------------------")

	relative_loss("res:", ref_out, cuda_out)
	relative_loss("points:", points_ref.grad, points_cuda.grad)
	relative_loss("points_p:", points_ref_p.grad, points_cuda_p.grad)

test_xfm_points()
test_xfm_vectors()
