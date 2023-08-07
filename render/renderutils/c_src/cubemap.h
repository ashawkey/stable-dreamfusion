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

struct DiffuseCubemapKernelParams
{
    Tensor  cubemap;
    Tensor  out;
    dim3    gridSize;
};

struct SpecularCubemapKernelParams
{
    Tensor  cubemap;
    Tensor  bounds;
    Tensor  out;
    dim3    gridSize;
    float   costheta_cutoff;
    float   roughness;
};

struct SpecularBoundsKernelParams
{
    float   costheta_cutoff;
    Tensor  out;
    dim3    gridSize;
};
