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

#include "common.h"
#include "normal.h"

#define NORMAL_THRESHOLD 0.1f

//------------------------------------------------------------------------
// Perturb shading normal by tangent frame

__device__ vec3f fwdPerturbNormal(const vec3f perturbed_nrm, const vec3f smooth_nrm, const vec3f smooth_tng, bool opengl)
{
    vec3f _smooth_bitng = cross(smooth_tng, smooth_nrm);
    vec3f smooth_bitng = safeNormalize(_smooth_bitng);
    vec3f _shading_nrm = smooth_tng * perturbed_nrm.x + (opengl ? -1 : 1) * smooth_bitng * perturbed_nrm.y + smooth_nrm * max(perturbed_nrm.z, 0.0f);
    return safeNormalize(_shading_nrm);
}

__device__ void bwdPerturbNormal(const vec3f perturbed_nrm, const vec3f smooth_nrm, const vec3f smooth_tng, vec3f &d_perturbed_nrm, vec3f &d_smooth_nrm, vec3f &d_smooth_tng, const vec3f d_out, bool opengl)
{
    ////////////////////////////////////////////////////////////////////////
    // FWD
    vec3f _smooth_bitng = cross(smooth_tng, smooth_nrm);
    vec3f smooth_bitng = safeNormalize(_smooth_bitng);
    vec3f _shading_nrm = smooth_tng * perturbed_nrm.x + (opengl ? -1 : 1) * smooth_bitng * perturbed_nrm.y + smooth_nrm * max(perturbed_nrm.z, 0.0f);
        
    ////////////////////////////////////////////////////////////////////////
    // BWD
    vec3f d_shading_nrm(0);
    bwdSafeNormalize(_shading_nrm, d_shading_nrm, d_out);

    vec3f d_smooth_bitng(0);
    
    if (perturbed_nrm.z > 0.0f)
    {
        d_smooth_nrm += d_shading_nrm * perturbed_nrm.z;
        d_perturbed_nrm.z += sum(d_shading_nrm * smooth_nrm);
    }

    d_smooth_bitng += (opengl ? -1 : 1) * d_shading_nrm * perturbed_nrm.y;
    d_perturbed_nrm.y += (opengl ? -1 : 1) * sum(d_shading_nrm * smooth_bitng);

    d_smooth_tng += d_shading_nrm * perturbed_nrm.x;
    d_perturbed_nrm.x += sum(d_shading_nrm * smooth_tng);

    vec3f d__smooth_bitng(0);
    bwdSafeNormalize(_smooth_bitng, d__smooth_bitng, d_smooth_bitng);

    bwdCross(smooth_tng, smooth_nrm, d_smooth_tng, d_smooth_nrm, d__smooth_bitng);
}

//------------------------------------------------------------------------
#define bent_nrm_eps 0.001f

__device__ vec3f fwdBendNormal(const vec3f view_vec, const vec3f smooth_nrm, const vec3f geom_nrm)
{
    float dp = dot(view_vec, smooth_nrm);
    float t = clamp(dp / NORMAL_THRESHOLD, 0.0f, 1.0f);
    return geom_nrm * (1.0f - t) + smooth_nrm * t;
}

__device__ void bwdBendNormal(const vec3f view_vec, const vec3f smooth_nrm, const vec3f geom_nrm, vec3f& d_view_vec, vec3f& d_smooth_nrm, vec3f& d_geom_nrm, const vec3f d_out)
{
    ////////////////////////////////////////////////////////////////////////
    // FWD
    float dp = dot(view_vec, smooth_nrm);
    float t = clamp(dp / NORMAL_THRESHOLD, 0.0f, 1.0f);

    ////////////////////////////////////////////////////////////////////////
    // BWD
    if (dp > NORMAL_THRESHOLD)
        d_smooth_nrm += d_out;
    else
    {
        // geom_nrm * (1.0f - t) + smooth_nrm * t;
        d_geom_nrm   += d_out * (1.0f - t);
        d_smooth_nrm += d_out * t;
        float d_t = sum(d_out * (smooth_nrm - geom_nrm));

        float d_dp = dp < 0.0f || dp > NORMAL_THRESHOLD ? 0.0f : d_t / NORMAL_THRESHOLD;

        bwdDot(view_vec, smooth_nrm, d_view_vec, d_smooth_nrm, d_dp);
    }
}

//------------------------------------------------------------------------
// Kernels

__global__ void PrepareShadingNormalFwdKernel(PrepareShadingNormalKernelParams p) 
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f pos = p.pos.fetch3(px, py, pz);
    vec3f view_pos = p.view_pos.fetch3(px, py, pz);
    vec3f perturbed_nrm = p.perturbed_nrm.fetch3(px, py, pz);
    vec3f _smooth_nrm = p.smooth_nrm.fetch3(px, py, pz);
    vec3f _smooth_tng = p.smooth_tng.fetch3(px, py, pz);
    vec3f geom_nrm = p.geom_nrm.fetch3(px, py, pz);

    vec3f smooth_nrm = safeNormalize(_smooth_nrm);
    vec3f smooth_tng = safeNormalize(_smooth_tng);
    vec3f view_vec = safeNormalize(view_pos - pos);
    vec3f shading_nrm = fwdPerturbNormal(perturbed_nrm, smooth_nrm, smooth_tng, p.opengl);

    vec3f res;
    if (p.two_sided_shading && dot(view_vec, geom_nrm) < 0.0f)
        res = fwdBendNormal(view_vec, -shading_nrm, -geom_nrm);
    else
        res = fwdBendNormal(view_vec, shading_nrm, geom_nrm);

    p.out.store(px, py, pz, res);
}

__global__ void PrepareShadingNormalBwdKernel(PrepareShadingNormalKernelParams p) 
{ 
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f pos = p.pos.fetch3(px, py, pz);
    vec3f view_pos = p.view_pos.fetch3(px, py, pz);
    vec3f perturbed_nrm = p.perturbed_nrm.fetch3(px, py, pz);
    vec3f _smooth_nrm = p.smooth_nrm.fetch3(px, py, pz);
    vec3f _smooth_tng = p.smooth_tng.fetch3(px, py, pz);
    vec3f geom_nrm = p.geom_nrm.fetch3(px, py, pz);
    vec3f d_out = p.out.fetch3(px, py, pz);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // FWD

    vec3f smooth_nrm = safeNormalize(_smooth_nrm);
    vec3f smooth_tng = safeNormalize(_smooth_tng);
    vec3f _view_vec = view_pos - pos;
    vec3f view_vec = safeNormalize(view_pos - pos);

    vec3f shading_nrm = fwdPerturbNormal(perturbed_nrm, smooth_nrm, smooth_tng, p.opengl);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // BWD

    vec3f d_view_vec(0), d_shading_nrm(0), d_geom_nrm(0);
    if (p.two_sided_shading && dot(view_vec, geom_nrm) < 0.0f)
    {
        bwdBendNormal(view_vec, -shading_nrm, -geom_nrm, d_view_vec, d_shading_nrm, d_geom_nrm, d_out);
        d_shading_nrm = -d_shading_nrm;
        d_geom_nrm = -d_geom_nrm;
    }
    else
        bwdBendNormal(view_vec, shading_nrm, geom_nrm, d_view_vec, d_shading_nrm, d_geom_nrm, d_out);

    vec3f d_perturbed_nrm(0), d_smooth_nrm(0), d_smooth_tng(0);
    bwdPerturbNormal(perturbed_nrm, smooth_nrm, smooth_tng, d_perturbed_nrm, d_smooth_nrm, d_smooth_tng, d_shading_nrm, p.opengl);

    vec3f d__view_vec(0), d__smooth_nrm(0), d__smooth_tng(0);
    bwdSafeNormalize(_view_vec, d__view_vec, d_view_vec);
    bwdSafeNormalize(_smooth_nrm, d__smooth_nrm, d_smooth_nrm);
    bwdSafeNormalize(_smooth_tng, d__smooth_tng, d_smooth_tng);

    p.pos.store_grad(px, py, pz, -d__view_vec);
    p.view_pos.store_grad(px, py, pz, d__view_vec);
    p.perturbed_nrm.store_grad(px, py, pz, d_perturbed_nrm);
    p.smooth_nrm.store_grad(px, py, pz, d__smooth_nrm);
    p.smooth_tng.store_grad(px, py, pz, d__smooth_tng);
    p.geom_nrm.store_grad(px, py, pz, d_geom_nrm);
}