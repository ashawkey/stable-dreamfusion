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

RES = 4
DTYPE = torch.float32

def relative_loss(name, ref, cuda):
	ref = ref.float()
	cuda = cuda.float()
	print(name, torch.max(torch.abs(ref - cuda) / torch.abs(ref + 1e-7)).item())

def test_cubemap():
	cubemap_cuda = torch.rand(6, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	cubemap_ref = cubemap_cuda.clone().detach().requires_grad_(True)
	weights = torch.rand(3, 3, 1, dtype=DTYPE, device='cuda')
	target = torch.rand(6, RES, RES, 3, dtype=DTYPE, device='cuda')

	ref = ru.filter_cubemap(cubemap_ref, weights, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.filter_cubemap(cubemap_cuda, weights, use_python=False)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Cubemap:")
	print("-------------------------------------------------------------")

	relative_loss("flt:", ref, cuda)
	relative_loss("cubemap:", cubemap_ref.grad, cubemap_cuda.grad)


test_cubemap()
