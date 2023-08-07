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

#pragma once

#include "common.h"

struct LambertKernelParams
{
    Tensor  nrm;
    Tensor  wi;
    Tensor  out;
    dim3    gridSize;
};

struct FrostbiteDiffuseKernelParams
{
    Tensor  nrm;
    Tensor  wi;
    Tensor  wo;
    Tensor  linearRoughness;
    Tensor  out;
    dim3    gridSize;
};

struct FresnelShlickKernelParams
{
    Tensor  f0;
    Tensor  f90;
    Tensor  cosTheta;
    Tensor  out;
    dim3    gridSize;
};

struct NdfGGXParams
{
    Tensor  alphaSqr;
    Tensor  cosTheta;
    Tensor  out;
    dim3    gridSize;
};

struct MaskingSmithParams
{
    Tensor  alphaSqr;
    Tensor  cosThetaI;
    Tensor  cosThetaO;
    Tensor  out;
    dim3    gridSize;
};

struct PbrSpecular
{
    Tensor  col;
    Tensor  nrm;
    Tensor  wo;
    Tensor  wi;
    Tensor  alpha;
    Tensor  out;
    dim3    gridSize;
    float   min_roughness;
};

struct PbrBSDF
{
    Tensor  kd;
    Tensor  arm;
    Tensor  pos;
    Tensor  nrm;
    Tensor  view_pos;
    Tensor  light_pos;
    Tensor  out;
    dim3    gridSize;
    float   min_roughness;
    int     BSDF;
};
