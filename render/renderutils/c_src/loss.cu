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

#include <cuda.h>

#include "common.h"
#include "loss.h"

//------------------------------------------------------------------------
// Utils

__device__ inline float bwdAbs(float x) { return x == 0.0f ? 0.0f : x < 0.0f ? -1.0f : 1.0f; }

__device__ float warpSum(float val) {
    for (int i = 1; i < 32; i *= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, i);
    return val;
}

//------------------------------------------------------------------------
// Tonemapping

__device__ inline float fwdSRGB(float x)
{
    return x > 0.0031308f ? powf(max(x, 0.0031308f), 1.0f / 2.4f) * 1.055f - 0.055f : 12.92f * max(x, 0.0f);
}

__device__ inline void bwdSRGB(float x, float &d_x, float d_out)
{
    if (x > 0.0031308f)
        d_x += d_out * 0.439583f / powf(x, 0.583333f);
    else if (x > 0.0f)
        d_x += d_out * 12.92f;
}

__device__ inline vec3f fwdTonemapLogSRGB(vec3f x)
{
    return vec3f(fwdSRGB(logf(x.x + 1.0f)), fwdSRGB(logf(x.y + 1.0f)), fwdSRGB(logf(x.z + 1.0f)));
}

__device__ inline void bwdTonemapLogSRGB(vec3f x, vec3f& d_x, vec3f d_out)
{
    if (x.x > 0.0f && x.x < 65535.0f)
    {
        bwdSRGB(logf(x.x + 1.0f), d_x.x, d_out.x);
        d_x.x *= 1 / (x.x + 1.0f);
    }
    if (x.y > 0.0f && x.y < 65535.0f)
    {
        bwdSRGB(logf(x.y + 1.0f), d_x.y, d_out.y);
        d_x.y *= 1 / (x.y + 1.0f);
    }
    if (x.z > 0.0f && x.z < 65535.0f)
    {
        bwdSRGB(logf(x.z + 1.0f), d_x.z, d_out.z);
        d_x.z *= 1 / (x.z + 1.0f);
    }
}

__device__ inline float fwdRELMSE(float img, float target, float eps = 0.1f)
{
    return (img - target) * (img - target) / (img * img + target * target + eps);
}

__device__ inline void bwdRELMSE(float img, float target, float &d_img, float &d_target, float d_out, float eps = 0.1f)
{
    float denom  = (target * target + img * img + eps);
    d_img    += d_out * 2 * (img - target) * (target * (target + img) + eps) / (denom * denom);
    d_target -= d_out * 2 * (img - target) * (img * (target + img) + eps) / (denom * denom);
}

__device__ inline float fwdSMAPE(float img, float target, float eps=0.01f)
{
    return abs(img - target) / (img + target + eps);
}

__device__ inline void bwdSMAPE(float img, float target, float& d_img, float& d_target, float d_out, float eps = 0.01f)
{
    float denom = (target + img + eps);
    d_img    += d_out * bwdAbs(img - target) * (2 * target + eps) / (denom * denom);
    d_target -= d_out * bwdAbs(img - target) * (2 * img + eps) / (denom * denom);
}

//------------------------------------------------------------------------
// Kernels

__global__ void imgLossFwdKernel(LossKernelParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;

    float floss = 0.0f;
    if (px < p.gridSize.x && py < p.gridSize.y && pz < p.gridSize.z)
    {
        vec3f img = p.img.fetch3(px, py, pz);
        vec3f target = p.target.fetch3(px, py, pz);

        img = vec3f(clamp(img.x, 0.0f, 65535.0f), clamp(img.y, 0.0f, 65535.0f), clamp(img.z, 0.0f, 65535.0f));
        target = vec3f(clamp(target.x, 0.0f, 65535.0f), clamp(target.y, 0.0f, 65535.0f), clamp(target.z, 0.0f, 65535.0f));

        if (p.tonemapper == TONEMAPPER_LOG_SRGB)
        {
            img = fwdTonemapLogSRGB(img);
            target = fwdTonemapLogSRGB(target);
        }

        vec3f vloss(0);
        if (p.loss == LOSS_MSE)
            vloss = (img - target) * (img - target);
        else if (p.loss == LOSS_RELMSE)
            vloss = vec3f(fwdRELMSE(img.x, target.x), fwdRELMSE(img.y, target.y), fwdRELMSE(img.z, target.z));
        else if (p.loss == LOSS_SMAPE)
            vloss = vec3f(fwdSMAPE(img.x, target.x), fwdSMAPE(img.y, target.y), fwdSMAPE(img.z, target.z));
        else
            vloss = vec3f(abs(img.x - target.x), abs(img.y - target.y), abs(img.z - target.z));
        
        floss = sum(vloss) / 3.0f;
    }

    floss = warpSum(floss);

    dim3 warpSize = getWarpSize(blockDim);
    if (px < p.gridSize.x && py < p.gridSize.y && pz < p.gridSize.z && threadIdx.x % warpSize.x == 0 && threadIdx.y % warpSize.y == 0 && threadIdx.z % warpSize.z == 0)
        p.out.store(px / warpSize.x, py / warpSize.y, pz / warpSize.z, floss);
}

__global__ void imgLossBwdKernel(LossKernelParams p)
{ 
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;

    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    dim3 warpSize = getWarpSize(blockDim);

    vec3f _img = p.img.fetch3(px, py, pz);
    vec3f _target = p.target.fetch3(px, py, pz);
    float d_out = p.out.fetch1(px / warpSize.x, py / warpSize.y, pz / warpSize.z);

    /////////////////////////////////////////////////////////////////////
    // FWD

    vec3f img = _img, target = _target;
    if (p.tonemapper == TONEMAPPER_LOG_SRGB)
    {
        img = fwdTonemapLogSRGB(img);
        target = fwdTonemapLogSRGB(target);
    }

    /////////////////////////////////////////////////////////////////////
    // BWD

    vec3f d_vloss = vec3f(d_out, d_out, d_out) / 3.0f;

    vec3f d_img(0), d_target(0);
    if (p.loss == LOSS_MSE)
    {
        d_img = vec3f(d_vloss.x * 2 * (img.x - target.x), d_vloss.y * 2 * (img.y - target.y), d_vloss.x * 2 * (img.z - target.z));
        d_target = -d_img;
    }
    else if (p.loss == LOSS_RELMSE)
    {
        bwdRELMSE(img.x, target.x, d_img.x, d_target.x, d_vloss.x);
        bwdRELMSE(img.y, target.y, d_img.y, d_target.y, d_vloss.y);
        bwdRELMSE(img.z, target.z, d_img.z, d_target.z, d_vloss.z);
    }
    else if (p.loss == LOSS_SMAPE)
    {
        bwdSMAPE(img.x, target.x, d_img.x, d_target.x, d_vloss.x);
        bwdSMAPE(img.y, target.y, d_img.y, d_target.y, d_vloss.y);
        bwdSMAPE(img.z, target.z, d_img.z, d_target.z, d_vloss.z);
    }
    else
    {
        d_img = d_vloss * vec3f(bwdAbs(img.x - target.x), bwdAbs(img.y - target.y), bwdAbs(img.z - target.z));
        d_target = -d_img;
    }


    if (p.tonemapper == TONEMAPPER_LOG_SRGB)
    {
        vec3f d__img(0), d__target(0);
        bwdTonemapLogSRGB(_img, d__img, d_img);
        bwdTonemapLogSRGB(_target, d__target, d_target);
        d_img = d__img; d_target = d__target;
    }

    if (_img.x <= 0.0f || _img.x >= 65535.0f) d_img.x = 0;
    if (_img.y <= 0.0f || _img.y >= 65535.0f) d_img.y = 0;
    if (_img.z <= 0.0f || _img.z >= 65535.0f) d_img.z = 0;
    if (_target.x <= 0.0f || _target.x >= 65535.0f) d_target.x = 0;
    if (_target.y <= 0.0f || _target.y >= 65535.0f) d_target.y = 0;
    if (_target.z <= 0.0f || _target.z >= 65535.0f) d_target.z = 0;

    p.img.store_grad(px, py, pz, d_img);
    p.target.store_grad(px, py, pz, d_target);
}