/*
 * Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related 
 * documentation and any modifications thereto. Any use, reproduction, 
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or 
 * its affiliates is strictly prohibited.
 */

#ifdef _MSC_VER 
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <string>

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) { cudaError_t err = CUDA_CALL; AT_CUDA_CHECK(cudaGetLastError()); }
#define NVDR_CHECK_GL_ERROR(GL_CALL) { GL_CALL; GLenum err = glGetError(); TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); }
#define CHECK_TENSOR(X, DIMS, CHANNELS) \
    TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor") \
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16") \
    TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions") \
    TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

#include "common.h"
#include "loss.h"
#include "normal.h"
#include "cubemap.h"
#include "bsdf.h"
#include "mesh.h"

#define BLOCK_X 8
#define BLOCK_Y 8

//------------------------------------------------------------------------
// mesh.cu

void xfmPointsFwdKernel(XfmKernelParams p);
void xfmPointsBwdKernel(XfmKernelParams p);

//------------------------------------------------------------------------
// loss.cu

void imgLossFwdKernel(LossKernelParams p);
void imgLossBwdKernel(LossKernelParams p);

//------------------------------------------------------------------------
// normal.cu

void PrepareShadingNormalFwdKernel(PrepareShadingNormalKernelParams p);
void PrepareShadingNormalBwdKernel(PrepareShadingNormalKernelParams p);

//------------------------------------------------------------------------
// cubemap.cu

void DiffuseCubemapFwdKernel(DiffuseCubemapKernelParams p);
void DiffuseCubemapBwdKernel(DiffuseCubemapKernelParams p);
void SpecularBoundsKernel(SpecularBoundsKernelParams p);
void SpecularCubemapFwdKernel(SpecularCubemapKernelParams p);
void SpecularCubemapBwdKernel(SpecularCubemapKernelParams p);

//------------------------------------------------------------------------
// bsdf.cu

void LambertFwdKernel(LambertKernelParams p);
void LambertBwdKernel(LambertKernelParams p);

void FrostbiteDiffuseFwdKernel(FrostbiteDiffuseKernelParams p);
void FrostbiteDiffuseBwdKernel(FrostbiteDiffuseKernelParams p);

void FresnelShlickFwdKernel(FresnelShlickKernelParams p);
void FresnelShlickBwdKernel(FresnelShlickKernelParams p);

void ndfGGXFwdKernel(NdfGGXParams p);
void ndfGGXBwdKernel(NdfGGXParams p);

void lambdaGGXFwdKernel(NdfGGXParams p);
void lambdaGGXBwdKernel(NdfGGXParams p);

void maskingSmithFwdKernel(MaskingSmithParams p);
void maskingSmithBwdKernel(MaskingSmithParams p);

void pbrSpecularFwdKernel(PbrSpecular p);
void pbrSpecularBwdKernel(PbrSpecular p);

void pbrBSDFFwdKernel(PbrBSDF p);
void pbrBSDFBwdKernel(PbrBSDF p);

//------------------------------------------------------------------------
// Tensor helpers

void update_grid(dim3 &gridSize, torch::Tensor x)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
}

template<typename... Ts>
void update_grid(dim3& gridSize, torch::Tensor x, Ts&&... vs)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
    update_grid(gridSize, std::forward<Ts>(vs)...);
}

Tensor make_cuda_tensor(torch::Tensor val)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    return res;
}

Tensor make_cuda_tensor(torch::Tensor val, dim3 outDims, torch::Tensor* grad = nullptr)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    if (val.dim() == 4)
        res._dims[0] = outDims.z, res._dims[1] = outDims.y, res._dims[2] = outDims.x, res._dims[3] = val.size(3);
    else
        res._dims[0] = outDims.z, res._dims[1] = outDims.x, res._dims[2] = val.size(2), res._dims[3] = 1; // Add a trailing one for indexing math to work out

    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    if (grad != nullptr)
    {
        if (val.dim() == 4)
            *grad = torch::empty({ outDims.z, outDims.y, outDims.x, val.size(3) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));
        else // 3
            *grad = torch::empty({ outDims.z, outDims.x, val.size(2) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));

        res.d_val = res.fp16 ? (void*)grad->data_ptr<torch::BFloat16>() : (void*)grad->data_ptr<float>();
    }
    return res;
}

//------------------------------------------------------------------------
// prepare_shading_normal

torch::Tensor prepare_shading_normal_fwd(torch::Tensor pos, torch::Tensor view_pos, torch::Tensor perturbed_nrm, torch::Tensor smooth_nrm, torch::Tensor smooth_tng, torch::Tensor geom_nrm, bool two_sided_shading, bool opengl, bool fp16)
{
    CHECK_TENSOR(pos, 4, 3);
    CHECK_TENSOR(view_pos, 4, 3);
    CHECK_TENSOR(perturbed_nrm, 4, 3);
    CHECK_TENSOR(smooth_nrm, 4, 3);
    CHECK_TENSOR(smooth_tng, 4, 3);
    CHECK_TENSOR(geom_nrm, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PrepareShadingNormalKernelParams p;
    p.two_sided_shading = two_sided_shading;
    p.opengl = opengl;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.pos = make_cuda_tensor(pos, p.gridSize);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize);
    p.perturbed_nrm = make_cuda_tensor(perturbed_nrm, p.gridSize);
    p.smooth_nrm = make_cuda_tensor(smooth_nrm, p.gridSize);
    p.smooth_tng = make_cuda_tensor(smooth_tng, p.gridSize);
    p.geom_nrm = make_cuda_tensor(geom_nrm, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)PrepareShadingNormalFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> prepare_shading_normal_bwd(torch::Tensor pos, torch::Tensor view_pos, torch::Tensor perturbed_nrm, torch::Tensor smooth_nrm, torch::Tensor smooth_tng, torch::Tensor geom_nrm, torch::Tensor grad, bool two_sided_shading, bool opengl)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PrepareShadingNormalKernelParams p;
    p.two_sided_shading = two_sided_shading;
    p.opengl = opengl;
    update_grid(p.gridSize, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor pos_grad, view_pos_grad, perturbed_nrm_grad, smooth_nrm_grad, smooth_tng_grad, geom_nrm_grad;
    p.pos = make_cuda_tensor(pos, p.gridSize, &pos_grad);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize, &view_pos_grad);
    p.perturbed_nrm = make_cuda_tensor(perturbed_nrm, p.gridSize, &perturbed_nrm_grad);
    p.smooth_nrm = make_cuda_tensor(smooth_nrm, p.gridSize, &smooth_nrm_grad);
    p.smooth_tng = make_cuda_tensor(smooth_tng, p.gridSize, &smooth_tng_grad);
    p.geom_nrm = make_cuda_tensor(geom_nrm, p.gridSize, &geom_nrm_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)PrepareShadingNormalBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(pos_grad, view_pos_grad, perturbed_nrm_grad, smooth_nrm_grad, smooth_tng_grad, geom_nrm_grad);
}

//------------------------------------------------------------------------
// lambert

torch::Tensor lambert_fwd(torch::Tensor nrm, torch::Tensor wi, bool fp16)
{
    CHECK_TENSOR(nrm, 4, 3);
    CHECK_TENSOR(wi, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    LambertKernelParams p;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, nrm, wi);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 1 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.nrm = make_cuda_tensor(nrm, p.gridSize);
    p.wi = make_cuda_tensor(wi, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)LambertFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> lambert_bwd(torch::Tensor nrm, torch::Tensor wi, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    LambertKernelParams p;
    update_grid(p.gridSize, nrm, wi);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor nrm_grad, wi_grad;
    p.nrm = make_cuda_tensor(nrm, p.gridSize, &nrm_grad);
    p.wi = make_cuda_tensor(wi, p.gridSize, &wi_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)LambertBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor>(nrm_grad, wi_grad);
}

//------------------------------------------------------------------------
// frostbite diffuse

torch::Tensor frostbite_fwd(torch::Tensor nrm, torch::Tensor wi, torch::Tensor wo, torch::Tensor linearRoughness, bool fp16)
{
    CHECK_TENSOR(nrm, 4, 3);
    CHECK_TENSOR(wi, 4, 3);
    CHECK_TENSOR(wo, 4, 3);
    CHECK_TENSOR(linearRoughness, 4, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    FrostbiteDiffuseKernelParams p;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, nrm, wi, wo, linearRoughness);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 1 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.nrm = make_cuda_tensor(nrm, p.gridSize);
    p.wi = make_cuda_tensor(wi, p.gridSize);
    p.wo = make_cuda_tensor(wo, p.gridSize);
    p.linearRoughness = make_cuda_tensor(linearRoughness, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)FrostbiteDiffuseFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> frostbite_bwd(torch::Tensor nrm, torch::Tensor wi, torch::Tensor wo, torch::Tensor linearRoughness, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    FrostbiteDiffuseKernelParams p;
    update_grid(p.gridSize, nrm, wi, wo, linearRoughness);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor nrm_grad, wi_grad, wo_grad, linearRoughness_grad;
    p.nrm = make_cuda_tensor(nrm, p.gridSize, &nrm_grad);
    p.wi = make_cuda_tensor(wi, p.gridSize, &wi_grad);
    p.wo = make_cuda_tensor(wo, p.gridSize, &wo_grad);
    p.linearRoughness = make_cuda_tensor(linearRoughness, p.gridSize, &linearRoughness_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)FrostbiteDiffuseBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(nrm_grad, wi_grad, wo_grad, linearRoughness_grad);
}

//------------------------------------------------------------------------
// fresnel_shlick

torch::Tensor fresnel_shlick_fwd(torch::Tensor f0, torch::Tensor f90, torch::Tensor cosTheta, bool fp16)
{
    CHECK_TENSOR(f0, 4, 3);
    CHECK_TENSOR(f90, 4, 3);
    CHECK_TENSOR(cosTheta, 4, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    FresnelShlickKernelParams p;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, f0, f90, cosTheta);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.f0 = make_cuda_tensor(f0, p.gridSize);
    p.f90 = make_cuda_tensor(f90, p.gridSize);
    p.cosTheta = make_cuda_tensor(cosTheta, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)FresnelShlickFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fresnel_shlick_bwd(torch::Tensor f0, torch::Tensor f90, torch::Tensor cosTheta, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    FresnelShlickKernelParams p;
    update_grid(p.gridSize, f0, f90, cosTheta);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor f0_grad, f90_grad, cosT_grad;
    p.f0 = make_cuda_tensor(f0, p.gridSize, &f0_grad);
    p.f90 = make_cuda_tensor(f90, p.gridSize, &f90_grad);
    p.cosTheta = make_cuda_tensor(cosTheta, p.gridSize, &cosT_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)FresnelShlickBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(f0_grad, f90_grad, cosT_grad);
}

//------------------------------------------------------------------------
// ndf_ggd

torch::Tensor ndf_ggx_fwd(torch::Tensor alphaSqr, torch::Tensor cosTheta, bool fp16)
{
    CHECK_TENSOR(alphaSqr, 4, 1);
    CHECK_TENSOR(cosTheta, 4, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    NdfGGXParams p;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, alphaSqr, cosTheta);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 1 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.alphaSqr = make_cuda_tensor(alphaSqr, p.gridSize);
    p.cosTheta = make_cuda_tensor(cosTheta, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)ndfGGXFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> ndf_ggx_bwd(torch::Tensor alphaSqr, torch::Tensor cosTheta, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    NdfGGXParams p;
    update_grid(p.gridSize, alphaSqr, cosTheta);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor alphaSqr_grad, cosTheta_grad;
    p.alphaSqr = make_cuda_tensor(alphaSqr, p.gridSize, &alphaSqr_grad);
    p.cosTheta = make_cuda_tensor(cosTheta, p.gridSize, &cosTheta_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)ndfGGXBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor>(alphaSqr_grad, cosTheta_grad);
}

//------------------------------------------------------------------------
// lambda_ggx

torch::Tensor lambda_ggx_fwd(torch::Tensor alphaSqr, torch::Tensor cosTheta, bool fp16)
{
    CHECK_TENSOR(alphaSqr, 4, 1);
    CHECK_TENSOR(cosTheta, 4, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    NdfGGXParams p;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, alphaSqr, cosTheta);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 1 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.alphaSqr = make_cuda_tensor(alphaSqr, p.gridSize);
    p.cosTheta = make_cuda_tensor(cosTheta, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)lambdaGGXFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> lambda_ggx_bwd(torch::Tensor alphaSqr, torch::Tensor cosTheta, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    NdfGGXParams p;
    update_grid(p.gridSize, alphaSqr, cosTheta);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor alphaSqr_grad, cosTheta_grad;
    p.alphaSqr = make_cuda_tensor(alphaSqr, p.gridSize, &alphaSqr_grad);
    p.cosTheta = make_cuda_tensor(cosTheta, p.gridSize, &cosTheta_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)lambdaGGXBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor>(alphaSqr_grad, cosTheta_grad);
}

//------------------------------------------------------------------------
// masking_smith

torch::Tensor masking_smith_fwd(torch::Tensor alphaSqr, torch::Tensor cosThetaI, torch::Tensor cosThetaO, bool fp16)
{
    CHECK_TENSOR(alphaSqr, 4, 1);
    CHECK_TENSOR(cosThetaI, 4, 1);
    CHECK_TENSOR(cosThetaO, 4, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    MaskingSmithParams p;
    p.out.fp16 = fp16;
    update_grid(p.gridSize, alphaSqr, cosThetaI, cosThetaO);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 1 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.alphaSqr = make_cuda_tensor(alphaSqr, p.gridSize);
    p.cosThetaI = make_cuda_tensor(cosThetaI, p.gridSize);
    p.cosThetaO = make_cuda_tensor(cosThetaO, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)maskingSmithFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> masking_smith_bwd(torch::Tensor alphaSqr, torch::Tensor cosThetaI, torch::Tensor cosThetaO, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    MaskingSmithParams p;
    update_grid(p.gridSize, alphaSqr, cosThetaI, cosThetaO);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor alphaSqr_grad, cosThetaI_grad, cosThetaO_grad;
    p.alphaSqr = make_cuda_tensor(alphaSqr, p.gridSize, &alphaSqr_grad);
    p.cosThetaI = make_cuda_tensor(cosThetaI, p.gridSize, &cosThetaI_grad);
    p.cosThetaO = make_cuda_tensor(cosThetaO, p.gridSize, &cosThetaO_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)maskingSmithBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(alphaSqr_grad, cosThetaI_grad, cosThetaO_grad);
}

//------------------------------------------------------------------------
// pbr_specular

torch::Tensor pbr_specular_fwd(torch::Tensor col, torch::Tensor nrm, torch::Tensor wo, torch::Tensor wi, torch::Tensor alpha, float min_roughness, bool fp16)
{
    CHECK_TENSOR(col, 4, 3);
    CHECK_TENSOR(nrm, 4, 3);
    CHECK_TENSOR(wo, 4, 3);
    CHECK_TENSOR(wi, 4, 3);
    CHECK_TENSOR(alpha, 4, 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PbrSpecular p;
    p.out.fp16 = fp16;
    p.min_roughness = min_roughness;
    update_grid(p.gridSize, col, nrm, wo, wi, alpha);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.col = make_cuda_tensor(col, p.gridSize);
    p.nrm = make_cuda_tensor(nrm, p.gridSize);
    p.wo = make_cuda_tensor(wo, p.gridSize);
    p.wi = make_cuda_tensor(wi, p.gridSize);
    p.alpha = make_cuda_tensor(alpha, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)pbrSpecularFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pbr_specular_bwd(torch::Tensor col, torch::Tensor nrm, torch::Tensor wo, torch::Tensor wi, torch::Tensor alpha, float min_roughness, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PbrSpecular p;
    update_grid(p.gridSize, col, nrm, wo, wi, alpha);
    p.min_roughness = min_roughness;

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor col_grad, nrm_grad, wo_grad, wi_grad, alpha_grad;
    p.col = make_cuda_tensor(col, p.gridSize, &col_grad);
    p.nrm = make_cuda_tensor(nrm, p.gridSize, &nrm_grad);
    p.wo = make_cuda_tensor(wo, p.gridSize, &wo_grad);
    p.wi = make_cuda_tensor(wi, p.gridSize, &wi_grad);
    p.alpha = make_cuda_tensor(alpha, p.gridSize, &alpha_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)pbrSpecularBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(col_grad, nrm_grad, wo_grad, wi_grad, alpha_grad);
}

//------------------------------------------------------------------------
// pbr_bsdf

torch::Tensor pbr_bsdf_fwd(torch::Tensor kd, torch::Tensor arm, torch::Tensor pos, torch::Tensor nrm, torch::Tensor view_pos, torch::Tensor light_pos, float min_roughness, int BSDF, bool fp16)
{
    CHECK_TENSOR(kd, 4, 3);
    CHECK_TENSOR(arm, 4, 3);
    CHECK_TENSOR(pos, 4, 3);
    CHECK_TENSOR(nrm, 4, 3);
    CHECK_TENSOR(view_pos, 4, 3);
    CHECK_TENSOR(light_pos, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PbrBSDF p;
    p.out.fp16 = fp16;
    p.min_roughness = min_roughness;
    p.BSDF = BSDF;
    update_grid(p.gridSize, kd, arm, pos, nrm, view_pos, light_pos);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    p.kd = make_cuda_tensor(kd, p.gridSize);
    p.arm = make_cuda_tensor(arm, p.gridSize);
    p.pos = make_cuda_tensor(pos, p.gridSize);
    p.nrm = make_cuda_tensor(nrm, p.gridSize);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize);
    p.light_pos = make_cuda_tensor(light_pos, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)pbrBSDFFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pbr_bsdf_bwd(torch::Tensor kd, torch::Tensor arm, torch::Tensor pos, torch::Tensor nrm, torch::Tensor view_pos, torch::Tensor light_pos, float min_roughness, int BSDF, torch::Tensor grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    PbrBSDF p;
    update_grid(p.gridSize, kd, arm, pos, nrm, view_pos, light_pos);
    p.min_roughness = min_roughness;
    p.BSDF = BSDF;

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor kd_grad, arm_grad, pos_grad, nrm_grad, view_pos_grad, light_pos_grad;
    p.kd = make_cuda_tensor(kd, p.gridSize, &kd_grad);
    p.arm = make_cuda_tensor(arm, p.gridSize, &arm_grad);
    p.pos = make_cuda_tensor(pos, p.gridSize, &pos_grad);
    p.nrm = make_cuda_tensor(nrm, p.gridSize, &nrm_grad);
    p.view_pos = make_cuda_tensor(view_pos, p.gridSize, &view_pos_grad);
    p.light_pos = make_cuda_tensor(light_pos, p.gridSize, &light_pos_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)pbrBSDFBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(kd_grad, arm_grad, pos_grad, nrm_grad, view_pos_grad, light_pos_grad);
}

//------------------------------------------------------------------------
// filter_cubemap

torch::Tensor diffuse_cubemap_fwd(torch::Tensor cubemap)
{
    CHECK_TENSOR(cubemap, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    DiffuseCubemapKernelParams p;
    update_grid(p.gridSize, cubemap);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 3 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)DiffuseCubemapFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor diffuse_cubemap_bwd(torch::Tensor cubemap, torch::Tensor grad)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(grad, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    DiffuseCubemapKernelParams p;
    update_grid(p.gridSize, cubemap);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor cubemap_grad;
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    cubemap_grad = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, cubemap.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    p.cubemap.d_val = (void*)cubemap_grad.data_ptr<float>();

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)DiffuseCubemapBwdKernel, gridSize, blockSize, args, 0, stream));

    return cubemap_grad;
}

torch::Tensor specular_bounds(int resolution, float costheta_cutoff)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    SpecularBoundsKernelParams p;
    p.costheta_cutoff = costheta_cutoff;
    p.gridSize = dim3(resolution, resolution, 6);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 6*4 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)SpecularBoundsKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor specular_cubemap_fwd(torch::Tensor cubemap, torch::Tensor bounds, float roughness, float costheta_cutoff)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(bounds, 4, 6*4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    SpecularCubemapKernelParams p;
    p.roughness = roughness;
    p.costheta_cutoff = costheta_cutoff;
    update_grid(p.gridSize, cubemap);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ p.gridSize.z, p.gridSize.y, p.gridSize.x, 4 }, opts);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.bounds = make_cuda_tensor(bounds, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)SpecularCubemapFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor specular_cubemap_bwd(torch::Tensor cubemap, torch::Tensor bounds, torch::Tensor grad, float roughness, float costheta_cutoff)
{
    CHECK_TENSOR(cubemap, 4, 3);
    CHECK_TENSOR(bounds, 4, 6*4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    SpecularCubemapKernelParams p;
    p.roughness = roughness;
    p.costheta_cutoff = costheta_cutoff;
    update_grid(p.gridSize, cubemap);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Setup tensors
    torch::Tensor cubemap_grad;
    p.cubemap = make_cuda_tensor(cubemap, p.gridSize);
    p.bounds = make_cuda_tensor(bounds, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    cubemap_grad = torch::zeros({ p.gridSize.z, p.gridSize.y, p.gridSize.x, cubemap.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    p.cubemap.d_val = (void*)cubemap_grad.data_ptr<float>();

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)SpecularCubemapBwdKernel, gridSize, blockSize, args, 0, stream));

    return cubemap_grad;
}

//------------------------------------------------------------------------
// loss function

LossType strToLoss(std::string str)
{
    if (str == "mse")
        return LOSS_MSE;
    else if (str == "relmse")
        return LOSS_RELMSE;
    else if (str == "smape")
        return LOSS_SMAPE;
    else
        return LOSS_L1;
}

torch::Tensor image_loss_fwd(torch::Tensor img, torch::Tensor target, std::string loss, std::string tonemapper, bool fp16)
{
    CHECK_TENSOR(img, 4, 3);
    CHECK_TENSOR(target, 4, 3);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    LossKernelParams p;
    p.out.fp16 = fp16;
    p.loss = strToLoss(loss);
    p.tonemapper = tonemapper == "log_srgb" ? TONEMAPPER_LOG_SRGB : TONEMAPPER_NONE;
    update_grid(p.gridSize, img, target);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({ (p.gridSize.z - 1)/ warpSize.z + 1, (p.gridSize.y - 1) / warpSize.y + 1, (p.gridSize.x - 1) / warpSize.x + 1, 1 }, opts);

    p.img = make_cuda_tensor(img, p.gridSize);
    p.target = make_cuda_tensor(target, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)imgLossFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor> image_loss_bwd(torch::Tensor img, torch::Tensor target, torch::Tensor grad, std::string loss, std::string tonemapper)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    LossKernelParams p;
    p.loss = strToLoss(loss);
    p.tonemapper = tonemapper == "log_srgb" ? TONEMAPPER_LOG_SRGB : TONEMAPPER_NONE;
    update_grid(p.gridSize, img, target);

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(BLOCK_X, BLOCK_Y, p.gridSize);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor img_grad, target_grad;
    p.img = make_cuda_tensor(img, p.gridSize, &img_grad);
    p.target = make_cuda_tensor(target, p.gridSize, &target_grad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)imgLossBwdKernel, gridSize, blockSize, args, 0, stream));

    return std::tuple<torch::Tensor, torch::Tensor>(img_grad, target_grad);
}

//------------------------------------------------------------------------
// transform function

torch::Tensor xfm_fwd(torch::Tensor points, torch::Tensor matrix, bool isPoints, bool fp16)
{
    CHECK_TENSOR(points, 3, 3);
    CHECK_TENSOR(matrix, 3, 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    XfmKernelParams p;
    p.out.fp16 = fp16;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = isPoints ? torch::empty({ matrix.size(0), points.size(1), 4 }, opts) : torch::empty({ matrix.size(0), points.size(1), 3 }, opts);

    p.points = make_cuda_tensor(points, p.gridSize);
    p.matrix = make_cuda_tensor(matrix, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)xfmPointsFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor xfm_bwd(torch::Tensor points, torch::Tensor matrix, torch::Tensor grad, bool isPoints)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    XfmKernelParams p;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor points_grad;
    p.points = make_cuda_tensor(points, p.gridSize, &points_grad);
    p.matrix = make_cuda_tensor(matrix, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)xfmPointsBwdKernel, gridSize, blockSize, args, 0, stream));

    return points_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_shading_normal_fwd", &prepare_shading_normal_fwd, "prepare_shading_normal_fwd");
    m.def("prepare_shading_normal_bwd", &prepare_shading_normal_bwd, "prepare_shading_normal_bwd");
    m.def("lambert_fwd", &lambert_fwd, "lambert_fwd");
    m.def("lambert_bwd", &lambert_bwd, "lambert_bwd");
    m.def("frostbite_fwd", &frostbite_fwd, "frostbite_fwd");
    m.def("frostbite_bwd", &frostbite_bwd, "frostbite_bwd");
    m.def("fresnel_shlick_fwd", &fresnel_shlick_fwd, "fresnel_shlick_fwd");
    m.def("fresnel_shlick_bwd", &fresnel_shlick_bwd, "fresnel_shlick_bwd");
    m.def("ndf_ggx_fwd", &ndf_ggx_fwd, "ndf_ggx_fwd");
    m.def("ndf_ggx_bwd", &ndf_ggx_bwd, "ndf_ggx_bwd");
    m.def("lambda_ggx_fwd", &lambda_ggx_fwd, "lambda_ggx_fwd");
    m.def("lambda_ggx_bwd", &lambda_ggx_bwd, "lambda_ggx_bwd");
    m.def("masking_smith_fwd", &masking_smith_fwd, "masking_smith_fwd");
    m.def("masking_smith_bwd", &masking_smith_bwd, "masking_smith_bwd");
    m.def("pbr_specular_fwd", &pbr_specular_fwd, "pbr_specular_fwd");
    m.def("pbr_specular_bwd", &pbr_specular_bwd, "pbr_specular_bwd");
    m.def("pbr_bsdf_fwd", &pbr_bsdf_fwd, "pbr_bsdf_fwd");
    m.def("pbr_bsdf_bwd", &pbr_bsdf_bwd, "pbr_bsdf_bwd");
    m.def("diffuse_cubemap_fwd", &diffuse_cubemap_fwd, "diffuse_cubemap_fwd");
    m.def("diffuse_cubemap_bwd", &diffuse_cubemap_bwd, "diffuse_cubemap_bwd");
    m.def("specular_bounds", &specular_bounds, "specular_bounds");
    m.def("specular_cubemap_fwd", &specular_cubemap_fwd, "specular_cubemap_fwd");
    m.def("specular_cubemap_bwd", &specular_cubemap_bwd, "specular_cubemap_bwd");
    m.def("image_loss_fwd", &image_loss_fwd, "image_loss_fwd");
    m.def("image_loss_bwd", &image_loss_bwd, "image_loss_bwd");
    m.def("xfm_fwd", &xfm_fwd, "xfm_fwd");
    m.def("xfm_bwd", &xfm_bwd, "xfm_bwd");
}