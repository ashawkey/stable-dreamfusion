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
#include "bsdf.h"

#define SPECULAR_EPSILON 1e-4f

//------------------------------------------------------------------------
// Lambert functions

__device__ inline float fwdLambert(const vec3f nrm, const vec3f wi)
{
    return max(dot(nrm, wi) / M_PI, 0.0f);
}

__device__ inline void bwdLambert(const vec3f nrm, const vec3f wi, vec3f& d_nrm, vec3f& d_wi, const float d_out)
{
    if (dot(nrm, wi) > 0.0f)
        bwdDot(nrm, wi, d_nrm, d_wi, d_out / M_PI);
}

//------------------------------------------------------------------------
// Fresnel Schlick 

__device__ inline float fwdFresnelSchlick(const float f0, const float f90, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = powf(1.0f - _cosTheta, 5.0f);
    return f0 * (1.0f - scale) + f90 * scale;
}

__device__ inline void bwdFresnelSchlick(const float f0, const float f90, const float cosTheta, float& d_f0, float& d_f90, float& d_cosTheta, const float d_out)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = pow(max(1.0f - _cosTheta, 0.0f), 5.0f);
    d_f0 += d_out * (1.0 - scale);
    d_f90 += d_out * scale;
    if (cosTheta >= SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
    {
        d_cosTheta += d_out * (f90 - f0) * -5.0f * powf(1.0f - cosTheta, 4.0f);
    }
}

__device__ inline vec3f fwdFresnelSchlick(const vec3f f0, const vec3f f90, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = powf(1.0f - _cosTheta, 5.0f);
    return f0 * (1.0f - scale) + f90 * scale;
}

__device__ inline void bwdFresnelSchlick(const vec3f f0, const vec3f f90, const float cosTheta, vec3f& d_f0, vec3f& d_f90, float& d_cosTheta, const vec3f d_out)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float scale = pow(max(1.0f - _cosTheta, 0.0f), 5.0f);
    d_f0 += d_out * (1.0 - scale);
    d_f90 += d_out * scale;
    if (cosTheta >= SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
    {
        d_cosTheta += sum(d_out * (f90 - f0) * -5.0f * powf(1.0f - cosTheta, 4.0f));
    }
}

//------------------------------------------------------------------------
// Frostbite diffuse

__device__ inline float fwdFrostbiteDiffuse(const vec3f nrm, const vec3f wi, const vec3f wo, float linearRoughness)
{
    float wiDotN = dot(wi, nrm);
    float woDotN = dot(wo, nrm);
    if (wiDotN > 0.0f && woDotN > 0.0f)
    {
        vec3f h = safeNormalize(wo + wi);
        float wiDotH = dot(wi, h);

        float energyBias = 0.5f * linearRoughness;
        float energyFactor = 1.0f - (0.51f / 1.51f) * linearRoughness;
        float f90 = energyBias + 2.f * wiDotH * wiDotH * linearRoughness;
        float f0 = 1.f;
        
        float wiScatter = fwdFresnelSchlick(f0, f90, wiDotN);
        float woScatter = fwdFresnelSchlick(f0, f90, woDotN);
        
        return wiScatter * woScatter * energyFactor;
    }
    else return 0.0f;
}

__device__ inline void bwdFrostbiteDiffuse(const vec3f nrm, const vec3f wi, const vec3f wo, float linearRoughness, vec3f& d_nrm, vec3f& d_wi, vec3f& d_wo, float &d_linearRoughness, const float d_out)
{
    float wiDotN = dot(wi, nrm);
    float woDotN = dot(wo, nrm);

    if (wiDotN > 0.0f && woDotN > 0.0f)
    {
        vec3f h = safeNormalize(wo + wi);
        float wiDotH = dot(wi, h);

        float energyBias = 0.5f * linearRoughness;
        float energyFactor = 1.0f - (0.51f / 1.51f) * linearRoughness;
        float f90 = energyBias + 2.f * wiDotH * wiDotH * linearRoughness;
        float f0 = 1.f;
        
        float wiScatter = fwdFresnelSchlick(f0, f90, wiDotN);
        float woScatter = fwdFresnelSchlick(f0, f90, woDotN);

        // -------------- BWD --------------
        // Backprop: return wiScatter * woScatter * energyFactor;
        float d_wiScatter = d_out * woScatter * energyFactor;
        float d_woScatter = d_out * wiScatter * energyFactor;
        float d_energyFactor = d_out * wiScatter * woScatter; 

        // Backprop: float woScatter = fwdFresnelSchlick(f0, f90, woDotN);
        float d_woDotN = 0.0f, d_f0 = 0.0, d_f90 = 0.0f;
        bwdFresnelSchlick(f0, f90, woDotN, d_f0, d_f90, d_woDotN, d_woScatter);

        // Backprop: float wiScatter = fwdFresnelSchlick(fd0, fd90, wiDotN);
        float d_wiDotN = 0.0f;
        bwdFresnelSchlick(f0, f90, wiDotN, d_f0, d_f90, d_wiDotN, d_wiScatter);

        // Backprop: float f90 = energyBias + 2.f * wiDotH * wiDotH * linearRoughness;
        float d_energyBias = d_f90;
        float d_wiDotH = d_f90 * 4 * wiDotH * linearRoughness;
        d_linearRoughness += d_f90 * 2 * wiDotH * wiDotH;

        // Backprop: float energyFactor = 1.0f - (0.51f / 1.51f) * linearRoughness;
        d_linearRoughness -= (0.51f / 1.51f) * d_energyFactor;

        // Backprop: float energyBias = 0.5f * linearRoughness;
        d_linearRoughness += 0.5 * d_energyBias;

        // Backprop: float wiDotH = dot(wi, h);
        vec3f d_h(0);
        bwdDot(wi, h, d_wi, d_h, d_wiDotH);

        // Backprop: vec3f h = safeNormalize(wo + wi);     
        vec3f d_wo_wi(0);
        bwdSafeNormalize(wo + wi, d_wo_wi, d_h);
        d_wi += d_wo_wi; d_wo += d_wo_wi;

        bwdDot(wo, nrm, d_wo, d_nrm, d_woDotN);
        bwdDot(wi, nrm, d_wi, d_nrm, d_wiDotN);
    }
}

//------------------------------------------------------------------------
// Ndf GGX

__device__ inline float fwdNdfGGX(const float alphaSqr, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1.0f;
    return alphaSqr / (d * d * M_PI);
}

__device__ inline void bwdNdfGGX(const float alphaSqr, const float cosTheta, float& d_alphaSqr, float& d_cosTheta, const float d_out)
{
    // Torch only back propagates if clamp doesn't trigger
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    d_alphaSqr += d_out * (1.0f - (alphaSqr + 1.0f) * cosThetaSqr) / (M_PI * powf((alphaSqr - 1.0) * cosThetaSqr + 1.0f, 3.0f));
    if (cosTheta > SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
    {
        d_cosTheta += d_out * -(4.0f * (alphaSqr - 1.0f) * alphaSqr * cosTheta) / (M_PI * powf((alphaSqr - 1.0) * cosThetaSqr + 1.0f, 3.0f));
    }
}

//------------------------------------------------------------------------
// Lambda GGX

__device__ inline float fwdLambdaGGX(const float alphaSqr, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    float tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr;
    float res = 0.5f * (sqrtf(1.0f + alphaSqr * tanThetaSqr) - 1.0f);
    return res;
}

__device__ inline void bwdLambdaGGX(const float alphaSqr, const float cosTheta, float& d_alphaSqr, float& d_cosTheta, const float d_out)
{
    float _cosTheta = clamp(cosTheta, SPECULAR_EPSILON, 1.0f - SPECULAR_EPSILON);
    float cosThetaSqr = _cosTheta * _cosTheta;
    float tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr;
    float res = 0.5f * (sqrtf(1.0f + alphaSqr * tanThetaSqr) - 1.0f);

    d_alphaSqr += d_out * (0.25 * tanThetaSqr) / sqrtf(alphaSqr * tanThetaSqr + 1.0f);
    if (cosTheta > SPECULAR_EPSILON && cosTheta < 1.0f - SPECULAR_EPSILON)
        d_cosTheta += d_out * -(0.5 * alphaSqr) / (powf(_cosTheta, 3.0f) * sqrtf(alphaSqr / cosThetaSqr - alphaSqr + 1.0f));
}

//------------------------------------------------------------------------
// Masking GGX

__device__ inline float fwdMaskingSmithGGXCorrelated(const float alphaSqr, const float cosThetaI, const float cosThetaO)
{
    float lambdaI = fwdLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = fwdLambdaGGX(alphaSqr, cosThetaO);
    return 1.0f / (1.0f + lambdaI + lambdaO);
}

__device__ inline void bwdMaskingSmithGGXCorrelated(const float alphaSqr, const float cosThetaI, const float cosThetaO, float& d_alphaSqr, float& d_cosThetaI, float& d_cosThetaO, const float d_out)
{
    // FWD eval
    float lambdaI = fwdLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = fwdLambdaGGX(alphaSqr, cosThetaO);

    // BWD eval
    float d_lambdaIO = -d_out / powf(1.0f + lambdaI + lambdaO, 2.0f);
    bwdLambdaGGX(alphaSqr, cosThetaI, d_alphaSqr, d_cosThetaI, d_lambdaIO);
    bwdLambdaGGX(alphaSqr, cosThetaO, d_alphaSqr, d_cosThetaO, d_lambdaIO);
}

//------------------------------------------------------------------------
// GGX specular

__device__ vec3f fwdPbrSpecular(const vec3f col, const vec3f nrm, const vec3f wo, const vec3f wi, const float alpha, const float min_roughness)
{
    float _alpha = clamp(alpha, min_roughness * min_roughness, 1.0f);
    float alphaSqr = _alpha * _alpha;

    vec3f h = safeNormalize(wo + wi);
    float woDotN = dot(wo, nrm);
    float wiDotN = dot(wi, nrm);
    float woDotH = dot(wo, h);
    float nDotH = dot(nrm, h);

    float D = fwdNdfGGX(alphaSqr, nDotH);
    float G = fwdMaskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN);
    vec3f F = fwdFresnelSchlick(col, 1.0f, woDotH);
    vec3f w = F * D * G * 0.25 / woDotN;

    bool frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON);
    return frontfacing ? w : 0.0f;
}

__device__ void bwdPbrSpecular(
    const vec3f col, const vec3f nrm, const vec3f wo, const vec3f wi, const float alpha, const float min_roughness,
    vec3f& d_col, vec3f& d_nrm, vec3f& d_wo, vec3f& d_wi, float& d_alpha, const vec3f d_out)
{
    ///////////////////////////////////////////////////////////////////////
    // FWD eval

    float _alpha = clamp(alpha, min_roughness * min_roughness, 1.0f);
    float alphaSqr = _alpha * _alpha;

    vec3f h = safeNormalize(wo + wi);
    float woDotN = dot(wo, nrm);
    float wiDotN = dot(wi, nrm);
    float woDotH = dot(wo, h);
    float nDotH = dot(nrm, h);

    float D = fwdNdfGGX(alphaSqr, nDotH);
    float G = fwdMaskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN);
    vec3f F = fwdFresnelSchlick(col, 1.0f, woDotH);
    vec3f w = F * D * G * 0.25 / woDotN;
    bool frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON);

    if (frontfacing)
    {
        ///////////////////////////////////////////////////////////////////////
        // BWD eval

        vec3f d_F = d_out * D * G * 0.25f / woDotN;
        float d_D = sum(d_out * F * G * 0.25f / woDotN);
        float d_G = sum(d_out * F * D * 0.25f / woDotN);

        float d_woDotN = -sum(d_out * F * D * G * 0.25f / (woDotN * woDotN));

        vec3f d_f90(0);
        float d_woDotH(0), d_wiDotN(0), d_nDotH(0), d_alphaSqr(0);
        bwdFresnelSchlick(col, 1.0f, woDotH, d_col, d_f90, d_woDotH, d_F);
        bwdMaskingSmithGGXCorrelated(alphaSqr, woDotN, wiDotN, d_alphaSqr, d_woDotN, d_wiDotN, d_G);
        bwdNdfGGX(alphaSqr, nDotH, d_alphaSqr, d_nDotH, d_D);

        vec3f d_h(0);
        bwdDot(nrm, h, d_nrm, d_h, d_nDotH);
        bwdDot(wo, h, d_wo, d_h, d_woDotH);
        bwdDot(wi, nrm, d_wi, d_nrm, d_wiDotN);
        bwdDot(wo, nrm, d_wo, d_nrm, d_woDotN);

        vec3f d_h_unnorm(0);
        bwdSafeNormalize(wo + wi, d_h_unnorm, d_h);
        d_wo += d_h_unnorm;
        d_wi += d_h_unnorm;

        if (alpha > min_roughness * min_roughness)
            d_alpha += d_alphaSqr * 2 * alpha;
    }
}

//------------------------------------------------------------------------
// Full PBR BSDF

__device__ vec3f fwdPbrBSDF(const vec3f kd, const vec3f arm, const vec3f pos, const vec3f nrm, const vec3f view_pos, const vec3f light_pos, const float min_roughness, int BSDF)
{
    vec3f wo = safeNormalize(view_pos - pos);
    vec3f wi = safeNormalize(light_pos - pos);

    float alpha = arm.y * arm.y;
    vec3f spec_col = (0.04f * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x);
    vec3f diff_col = kd * (1.0f - arm.z);

    float diff = 0.0f;
    if (BSDF == 0)
        diff = fwdLambert(nrm, wi);
    else
        diff = fwdFrostbiteDiffuse(nrm, wi, wo, arm.y);    
    vec3f diffuse = diff_col * diff;
    vec3f specular = fwdPbrSpecular(spec_col, nrm, wo, wi, alpha, min_roughness);

    return diffuse + specular;
}

__device__ void bwdPbrBSDF(
    const vec3f kd, const vec3f arm, const vec3f pos, const vec3f nrm, const vec3f view_pos, const vec3f light_pos, const float min_roughness, int BSDF,
    vec3f& d_kd, vec3f& d_arm, vec3f& d_pos, vec3f& d_nrm, vec3f& d_view_pos, vec3f& d_light_pos, const vec3f d_out)
{
    ////////////////////////////////////////////////////////////////////////
    // FWD
    vec3f _wi = light_pos - pos;
    vec3f _wo = view_pos - pos;
    vec3f wi = safeNormalize(_wi);
    vec3f wo = safeNormalize(_wo);

    float alpha = arm.y * arm.y;
    vec3f spec_col = (0.04f * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x);
    vec3f diff_col = kd * (1.0f - arm.z);
    float diff = 0.0f;
    if (BSDF == 0)
        diff = fwdLambert(nrm, wi);
    else
        diff = fwdFrostbiteDiffuse(nrm, wi, wo, arm.y);    

    ////////////////////////////////////////////////////////////////////////
    // BWD

    float d_alpha(0);
    vec3f d_spec_col(0), d_wi(0), d_wo(0);
    bwdPbrSpecular(spec_col, nrm, wo, wi, alpha, min_roughness, d_spec_col, d_nrm, d_wo, d_wi, d_alpha, d_out);

    float d_diff = sum(diff_col * d_out);
    if (BSDF == 0)
        bwdLambert(nrm, wi, d_nrm, d_wi, d_diff);
    else
        bwdFrostbiteDiffuse(nrm, wi, wo, arm.y, d_nrm, d_wi, d_wo, d_arm.y, d_diff);    

    // Backprop: diff_col = kd * (1.0f - arm.z)
    vec3f d_diff_col = d_out * diff;
    d_kd += d_diff_col * (1.0f - arm.z);
    d_arm.z -= sum(d_diff_col * kd);

    // Backprop: spec_col = (0.04f * (1.0f - arm.z) + kd * arm.z) * (1.0 - arm.x)
    d_kd -= d_spec_col * (arm.x - 1.0f) * arm.z;
    d_arm.x += sum(d_spec_col * (arm.z * (0.04f - kd) - 0.04f));
    d_arm.z -= sum(d_spec_col * (kd - 0.04f) * (arm.x - 1.0f));

    // Backprop: alpha = arm.y * arm.y
    d_arm.y += d_alpha * 2 * arm.y;

    // Backprop: vec3f wi = safeNormalize(light_pos - pos);
    vec3f d__wi(0);
    bwdSafeNormalize(_wi, d__wi, d_wi);
    d_light_pos += d__wi;
    d_pos -= d__wi;

    // Backprop: vec3f wo = safeNormalize(view_pos - pos);
    vec3f d__wo(0);
    bwdSafeNormalize(_wo, d__wo, d_wo);
    d_view_pos += d__wo;
    d_pos -= d__wo;
}

//------------------------------------------------------------------------
// Kernels

__global__ void LambertFwdKernel(LambertKernelParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f wi = p.wi.fetch3(px, py, pz);

    float res = fwdLambert(nrm, wi);

    p.out.store(px, py, pz, res);
}

__global__ void LambertBwdKernel(LambertKernelParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f wi = p.wi.fetch3(px, py, pz);
    float d_out = p.out.fetch1(px, py, pz);

    vec3f d_nrm(0), d_wi(0);
    bwdLambert(nrm, wi, d_nrm, d_wi, d_out);

    p.nrm.store_grad(px, py, pz, d_nrm);
    p.wi.store_grad(px, py, pz, d_wi);
}

__global__ void FrostbiteDiffuseFwdKernel(FrostbiteDiffuseKernelParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f wi = p.wi.fetch3(px, py, pz);
    vec3f wo = p.wo.fetch3(px, py, pz);
    float linearRoughness = p.linearRoughness.fetch1(px, py, pz);

    float res = fwdFrostbiteDiffuse(nrm, wi, wo, linearRoughness);

    p.out.store(px, py, pz, res);
}

__global__ void FrostbiteDiffuseBwdKernel(FrostbiteDiffuseKernelParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f wi = p.wi.fetch3(px, py, pz);
    vec3f wo = p.wo.fetch3(px, py, pz);
    float linearRoughness = p.linearRoughness.fetch1(px, py, pz);
    float d_out = p.out.fetch1(px, py, pz);

    float d_linearRoughness = 0.0f;
    vec3f d_nrm(0), d_wi(0), d_wo(0);
    bwdFrostbiteDiffuse(nrm, wi, wo, linearRoughness, d_nrm, d_wi, d_wo, d_linearRoughness, d_out);

    p.nrm.store_grad(px, py, pz, d_nrm);
    p.wi.store_grad(px, py, pz, d_wi);
    p.wo.store_grad(px, py, pz, d_wo);
    p.linearRoughness.store_grad(px, py, pz, d_linearRoughness);
}

__global__ void FresnelShlickFwdKernel(FresnelShlickKernelParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f f0 = p.f0.fetch3(px, py, pz);
    vec3f f90 = p.f90.fetch3(px, py, pz);
    float cosTheta = p.cosTheta.fetch1(px, py, pz);

    vec3f res = fwdFresnelSchlick(f0, f90, cosTheta);
    p.out.store(px, py, pz, res);
}

__global__ void FresnelShlickBwdKernel(FresnelShlickKernelParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f f0 = p.f0.fetch3(px, py, pz);
    vec3f f90 = p.f90.fetch3(px, py, pz);
    float cosTheta = p.cosTheta.fetch1(px, py, pz);
    vec3f d_out = p.out.fetch3(px, py, pz);

    vec3f d_f0(0), d_f90(0);
    float d_cosTheta(0);
    bwdFresnelSchlick(f0, f90, cosTheta, d_f0, d_f90, d_cosTheta, d_out);

    p.f0.store_grad(px, py, pz, d_f0);
    p.f90.store_grad(px, py, pz, d_f90);
    p.cosTheta.store_grad(px, py, pz, d_cosTheta);
}

__global__ void ndfGGXFwdKernel(NdfGGXParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    float alphaSqr = p.alphaSqr.fetch1(px, py, pz);
    float cosTheta = p.cosTheta.fetch1(px, py, pz);
    float res = fwdNdfGGX(alphaSqr, cosTheta);
    
    p.out.store(px, py, pz, res);
}

__global__ void ndfGGXBwdKernel(NdfGGXParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    float alphaSqr = p.alphaSqr.fetch1(px, py, pz);
    float cosTheta = p.cosTheta.fetch1(px, py, pz);
    float d_out = p.out.fetch1(px, py, pz);

    float d_alphaSqr(0), d_cosTheta(0);
    bwdNdfGGX(alphaSqr, cosTheta, d_alphaSqr, d_cosTheta, d_out);

    p.alphaSqr.store_grad(px, py, pz, d_alphaSqr);
    p.cosTheta.store_grad(px, py, pz, d_cosTheta);
}

__global__ void lambdaGGXFwdKernel(NdfGGXParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    float alphaSqr = p.alphaSqr.fetch1(px, py, pz);
    float cosTheta = p.cosTheta.fetch1(px, py, pz);
    float res = fwdLambdaGGX(alphaSqr, cosTheta);

    p.out.store(px, py, pz, res);
}

__global__ void lambdaGGXBwdKernel(NdfGGXParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    float alphaSqr = p.alphaSqr.fetch1(px, py, pz);
    float cosTheta = p.cosTheta.fetch1(px, py, pz);
    float d_out = p.out.fetch1(px, py, pz);

    float d_alphaSqr(0), d_cosTheta(0);
    bwdLambdaGGX(alphaSqr, cosTheta, d_alphaSqr, d_cosTheta, d_out);

    p.alphaSqr.store_grad(px, py, pz, d_alphaSqr);
    p.cosTheta.store_grad(px, py, pz, d_cosTheta);
}

__global__ void maskingSmithFwdKernel(MaskingSmithParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    float alphaSqr = p.alphaSqr.fetch1(px, py, pz);
    float cosThetaI = p.cosThetaI.fetch1(px, py, pz);
    float cosThetaO = p.cosThetaO.fetch1(px, py, pz);
    float res = fwdMaskingSmithGGXCorrelated(alphaSqr, cosThetaI, cosThetaO);
    
    p.out.store(px, py, pz, res);
}

__global__ void maskingSmithBwdKernel(MaskingSmithParams p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    float alphaSqr = p.alphaSqr.fetch1(px, py, pz);
    float cosThetaI = p.cosThetaI.fetch1(px, py, pz);
    float cosThetaO = p.cosThetaO.fetch1(px, py, pz);
    float d_out = p.out.fetch1(px, py, pz);

    float d_alphaSqr(0), d_cosThetaI(0), d_cosThetaO(0);
    bwdMaskingSmithGGXCorrelated(alphaSqr, cosThetaI, cosThetaO, d_alphaSqr, d_cosThetaI, d_cosThetaO, d_out);

    p.alphaSqr.store_grad(px, py, pz, d_alphaSqr);
    p.cosThetaI.store_grad(px, py, pz, d_cosThetaI);
    p.cosThetaO.store_grad(px, py, pz, d_cosThetaO);
}

__global__ void pbrSpecularFwdKernel(PbrSpecular p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f col = p.col.fetch3(px, py, pz);
    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f wo = p.wo.fetch3(px, py, pz);
    vec3f wi = p.wi.fetch3(px, py, pz);
    float alpha = p.alpha.fetch1(px, py, pz);

    vec3f res = fwdPbrSpecular(col, nrm, wo, wi, alpha, p.min_roughness);

    p.out.store(px, py, pz, res);
}

__global__ void pbrSpecularBwdKernel(PbrSpecular p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f col = p.col.fetch3(px, py, pz);
    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f wo = p.wo.fetch3(px, py, pz);
    vec3f wi = p.wi.fetch3(px, py, pz);
    float alpha = p.alpha.fetch1(px, py, pz);
    vec3f d_out = p.out.fetch3(px, py, pz);

    float d_alpha(0);
    vec3f d_col(0), d_nrm(0), d_wo(0), d_wi(0);
    bwdPbrSpecular(col, nrm, wo, wi, alpha, p.min_roughness, d_col, d_nrm, d_wo, d_wi, d_alpha, d_out);

    p.col.store_grad(px, py, pz, d_col);
    p.nrm.store_grad(px, py, pz, d_nrm);
    p.wo.store_grad(px, py, pz, d_wo);
    p.wi.store_grad(px, py, pz, d_wi);
    p.alpha.store_grad(px, py, pz, d_alpha);
}

__global__ void pbrBSDFFwdKernel(PbrBSDF p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f kd = p.kd.fetch3(px, py, pz);
    vec3f arm = p.arm.fetch3(px, py, pz);
    vec3f pos = p.pos.fetch3(px, py, pz);
    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f view_pos = p.view_pos.fetch3(px, py, pz);
    vec3f light_pos = p.light_pos.fetch3(px, py, pz);

    vec3f res = fwdPbrBSDF(kd, arm, pos, nrm, view_pos, light_pos, p.min_roughness, p.BSDF);

    p.out.store(px, py, pz, res);
}
__global__ void pbrBSDFBwdKernel(PbrBSDF p)
{
    // Calculate pixel position.
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    vec3f kd = p.kd.fetch3(px, py, pz);
    vec3f arm = p.arm.fetch3(px, py, pz);
    vec3f pos = p.pos.fetch3(px, py, pz);
    vec3f nrm = p.nrm.fetch3(px, py, pz);
    vec3f view_pos = p.view_pos.fetch3(px, py, pz);
    vec3f light_pos = p.light_pos.fetch3(px, py, pz);
    vec3f d_out = p.out.fetch3(px, py, pz);

    vec3f d_kd(0), d_arm(0), d_pos(0), d_nrm(0), d_view_pos(0), d_light_pos(0);
    bwdPbrBSDF(kd, arm, pos, nrm, view_pos, light_pos, p.min_roughness, p.BSDF, d_kd, d_arm, d_pos, d_nrm, d_view_pos, d_light_pos, d_out);

    p.kd.store_grad(px, py, pz, d_kd);
    p.arm.store_grad(px, py, pz, d_arm);
    p.pos.store_grad(px, py, pz, d_pos);
    p.nrm.store_grad(px, py, pz, d_nrm);
    p.view_pos.store_grad(px, py, pz, d_view_pos);
    p.light_pos.store_grad(px, py, pz, d_light_pos);
}
