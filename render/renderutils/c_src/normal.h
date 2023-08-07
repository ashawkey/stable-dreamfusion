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

struct PrepareShadingNormalKernelParams
{
    Tensor  pos;
    Tensor  view_pos;
    Tensor  perturbed_nrm;
    Tensor  smooth_nrm;
    Tensor  smooth_tng;
    Tensor  geom_nrm;
    Tensor  out;
    dim3    gridSize;
    bool    two_sided_shading, opengl;
};
