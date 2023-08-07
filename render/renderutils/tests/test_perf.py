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

DTYPE=torch.float32

def test_bsdf(BATCH, RES, ITR):
	kd_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	kd_ref = kd_cuda.clone().detach().requires_grad_(True)
	arm_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	arm_ref = arm_cuda.clone().detach().requires_grad_(True)
	pos_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	pos_ref = pos_cuda.clone().detach().requires_grad_(True)
	nrm_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	nrm_ref = nrm_cuda.clone().detach().requires_grad_(True)
	view_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	view_ref = view_cuda.clone().detach().requires_grad_(True)
	light_cuda = torch.rand(BATCH, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	light_ref = light_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(BATCH, RES, RES, 3, device='cuda')

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	ru.pbr_bsdf(kd_cuda, arm_cuda, pos_cuda, nrm_cuda, view_cuda, light_cuda)

	print("--- Testing: [%d, %d, %d] ---" % (BATCH, RES, RES))

	start.record()
	for i in range(ITR):
		ref = ru.pbr_bsdf(kd_ref, arm_ref, pos_ref, nrm_ref, view_ref, light_ref, use_python=True)
	end.record()
	torch.cuda.synchronize()
	print("Pbr BSDF python:", start.elapsed_time(end))

	start.record()
	for i in range(ITR):
		cuda = ru.pbr_bsdf(kd_cuda, arm_cuda, pos_cuda, nrm_cuda, view_cuda, light_cuda)
	end.record()
	torch.cuda.synchronize()
	print("Pbr BSDF cuda:", start.elapsed_time(end))

test_bsdf(1, 512, 1000)
test_bsdf(16, 512, 1000)
test_bsdf(1, 2048, 1000)
