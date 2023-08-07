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
#include <cuda.h>
#include <stdint.h>

#include "vec3f.h"
#include "vec4f.h"
#include "tensor.h"

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, dim3 dims);
dim3 getLaunchGridSize(dim3 blockSize, dim3 dims);

#ifdef __CUDACC__

#ifdef _MSC_VER
#define M_PI 3.14159265358979323846f
#endif

__host__ __device__ static inline dim3 getWarpSize(dim3 blockSize)
{
    return dim3(
        min(blockSize.x, 32u),
        min(max(32u / blockSize.x, 1u), min(32u, blockSize.y)),
        min(max(32u / (blockSize.x * blockSize.y), 1u), min(32u, blockSize.z))
    );
}

__device__ static inline float clamp(float val, float mn, float mx) { return min(max(val, mn), mx); }
#else
dim3 getWarpSize(dim3 blockSize);
#endif