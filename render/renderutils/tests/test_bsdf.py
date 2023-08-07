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

def test_normal():
	pos_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	pos_ref = pos_cuda.clone().detach().requires_grad_(True)
	view_pos_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	view_pos_ref = view_pos_cuda.clone().detach().requires_grad_(True)
	perturbed_nrm_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	perturbed_nrm_ref = perturbed_nrm_cuda.clone().detach().requires_grad_(True)
	smooth_nrm_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	smooth_nrm_ref = smooth_nrm_cuda.clone().detach().requires_grad_(True)
	smooth_tng_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	smooth_tng_ref = smooth_tng_cuda.clone().detach().requires_grad_(True)
	geom_nrm_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	geom_nrm_ref = geom_nrm_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda')

	ref = ru.prepare_shading_normal(pos_ref, view_pos_ref, perturbed_nrm_ref, smooth_nrm_ref, smooth_tng_ref, geom_nrm_ref, True, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.prepare_shading_normal(pos_cuda, view_pos_cuda, perturbed_nrm_cuda, smooth_nrm_cuda, smooth_tng_cuda, geom_nrm_cuda, True)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    bent normal")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("pos:", pos_ref.grad, pos_cuda.grad)
	relative_loss("view_pos:", view_pos_ref.grad, view_pos_cuda.grad)
	relative_loss("perturbed_nrm:", perturbed_nrm_ref.grad, perturbed_nrm_cuda.grad)
	relative_loss("smooth_nrm:", smooth_nrm_ref.grad, smooth_nrm_cuda.grad)
	relative_loss("smooth_tng:", smooth_tng_ref.grad, smooth_tng_cuda.grad)
	relative_loss("geom_nrm:", geom_nrm_ref.grad, geom_nrm_cuda.grad)

def test_schlick():
	f0_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	f0_ref = f0_cuda.clone().detach().requires_grad_(True)
	f90_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	f90_ref = f90_cuda.clone().detach().requires_grad_(True)
	cosT_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True) * 2.0
	cosT_cuda = cosT_cuda.clone().detach().requires_grad_(True)
	cosT_ref = cosT_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda')

	ref = ru._fresnel_shlick(f0_ref, f90_ref, cosT_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru._fresnel_shlick(f0_cuda, f90_cuda, cosT_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Fresnel shlick")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("f0:", f0_ref.grad, f0_cuda.grad)
	relative_loss("f90:", f90_ref.grad, f90_cuda.grad)
	relative_loss("cosT:", cosT_ref.grad, cosT_cuda.grad)

def test_ndf_ggx():
	alphaSqr_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True)
	alphaSqr_cuda = alphaSqr_cuda.clone().detach().requires_grad_(True)
	alphaSqr_ref = alphaSqr_cuda.clone().detach().requires_grad_(True)
	cosT_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True) * 3.0 - 1
	cosT_cuda = cosT_cuda.clone().detach().requires_grad_(True)
	cosT_ref = cosT_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda')

	ref = ru._ndf_ggx(alphaSqr_ref, cosT_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru._ndf_ggx(alphaSqr_cuda, cosT_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Ndf GGX")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("alpha:", alphaSqr_ref.grad, alphaSqr_cuda.grad)
	relative_loss("cosT:", cosT_ref.grad, cosT_cuda.grad)

def test_lambda_ggx():
	alphaSqr_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True)
	alphaSqr_ref = alphaSqr_cuda.clone().detach().requires_grad_(True)
	cosT_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True) * 3.0 - 1
	cosT_cuda = cosT_cuda.clone().detach().requires_grad_(True)
	cosT_ref = cosT_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda')

	ref = ru._lambda_ggx(alphaSqr_ref, cosT_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru._lambda_ggx(alphaSqr_cuda, cosT_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Lambda GGX")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("alpha:", alphaSqr_ref.grad, alphaSqr_cuda.grad)
	relative_loss("cosT:", cosT_ref.grad, cosT_cuda.grad)

def test_masking_smith():
	alphaSqr_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True)
	alphaSqr_ref = alphaSqr_cuda.clone().detach().requires_grad_(True)
	cosThetaI_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True)
	cosThetaI_ref = cosThetaI_cuda.clone().detach().requires_grad_(True)
	cosThetaO_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True)
	cosThetaO_ref = cosThetaO_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda')

	ref = ru._masking_smith(alphaSqr_ref, cosThetaI_ref, cosThetaO_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru._masking_smith(alphaSqr_cuda, cosThetaI_cuda, cosThetaO_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Smith masking term")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("alpha:", alphaSqr_ref.grad, alphaSqr_cuda.grad)
	relative_loss("cosThetaI:", cosThetaI_ref.grad, cosThetaI_cuda.grad)
	relative_loss("cosThetaO:", cosThetaO_ref.grad, cosThetaO_cuda.grad)

def test_lambert():
	normals_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	normals_ref = normals_cuda.clone().detach().requires_grad_(True)
	wi_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	wi_ref = wi_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda')

	ref = ru.lambert(normals_ref, wi_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.lambert(normals_cuda, wi_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Lambert")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("nrm:", normals_ref.grad, normals_cuda.grad)
	relative_loss("wi:", wi_ref.grad, wi_cuda.grad)

def test_frostbite():
	normals_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	normals_ref = normals_cuda.clone().detach().requires_grad_(True)
	wi_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	wi_ref = wi_cuda.clone().detach().requires_grad_(True)
	wo_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	wo_ref = wo_cuda.clone().detach().requires_grad_(True)
	rough_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True)
	rough_ref = rough_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda')

	ref = ru.frostbite_diffuse(normals_ref, wi_ref, wo_ref, rough_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.frostbite_diffuse(normals_cuda, wi_cuda, wo_cuda, rough_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Frostbite")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("nrm:", normals_ref.grad, normals_cuda.grad)
	relative_loss("wo:", wo_ref.grad, wo_cuda.grad)
	relative_loss("wi:", wi_ref.grad, wi_cuda.grad)
	relative_loss("rough:", rough_ref.grad, rough_cuda.grad)

def test_pbr_specular():
	col_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	col_ref = col_cuda.clone().detach().requires_grad_(True)
	nrm_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	nrm_ref = nrm_cuda.clone().detach().requires_grad_(True)
	wi_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	wi_ref = wi_cuda.clone().detach().requires_grad_(True)
	wo_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	wo_ref = wo_cuda.clone().detach().requires_grad_(True)
	alpha_cuda = torch.rand(1, RES, RES, 1, dtype=DTYPE, device='cuda', requires_grad=True)
	alpha_ref = alpha_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda')

	ref = ru.pbr_specular(col_ref, nrm_ref, wo_ref, wi_ref, alpha_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.pbr_specular(col_cuda, nrm_cuda, wo_cuda, wi_cuda, alpha_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Pbr specular")
	print("-------------------------------------------------------------")

	relative_loss("res:", ref, cuda)
	if col_ref.grad is not None:
		relative_loss("col:", col_ref.grad, col_cuda.grad)
	if nrm_ref.grad is not None:
		relative_loss("nrm:", nrm_ref.grad, nrm_cuda.grad)
	if wi_ref.grad is not None:
		relative_loss("wi:", wi_ref.grad, wi_cuda.grad)
	if wo_ref.grad is not None:
		relative_loss("wo:", wo_ref.grad, wo_cuda.grad)
	if alpha_ref.grad is not None:
		relative_loss("alpha:", alpha_ref.grad, alpha_cuda.grad)

def test_pbr_bsdf(bsdf):
	kd_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	kd_ref = kd_cuda.clone().detach().requires_grad_(True)
	arm_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	arm_ref = arm_cuda.clone().detach().requires_grad_(True)
	pos_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	pos_ref = pos_cuda.clone().detach().requires_grad_(True)
	nrm_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	nrm_ref = nrm_cuda.clone().detach().requires_grad_(True)
	view_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	view_ref = view_cuda.clone().detach().requires_grad_(True)
	light_cuda = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
	light_ref = light_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 3, dtype=DTYPE, device='cuda')

	ref = ru.pbr_bsdf(kd_ref, arm_ref, pos_ref, nrm_ref, view_ref, light_ref, use_python=True, bsdf=bsdf)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.pbr_bsdf(kd_cuda, arm_cuda, pos_cuda, nrm_cuda, view_cuda, light_cuda, bsdf=bsdf)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Pbr BSDF")
	print("-------------------------------------------------------------")

	relative_loss("res:", ref, cuda)
	if kd_ref.grad is not None:
		relative_loss("kd:", kd_ref.grad, kd_cuda.grad)
	if arm_ref.grad is not None:
		relative_loss("arm:", arm_ref.grad, arm_cuda.grad)
	if pos_ref.grad is not None:
		relative_loss("pos:", pos_ref.grad, pos_cuda.grad)
	if nrm_ref.grad is not None:
		relative_loss("nrm:", nrm_ref.grad, nrm_cuda.grad)
	if view_ref.grad is not None:
		relative_loss("view:", view_ref.grad, view_cuda.grad)
	if light_ref.grad is not None:
		relative_loss("light:", light_ref.grad, light_cuda.grad)

test_normal()

test_schlick()
test_ndf_ggx()
test_lambda_ggx()
test_masking_smith()

test_lambert()
test_frostbite()
test_pbr_specular()
test_pbr_bsdf('lambert')
test_pbr_bsdf('frostbite')
