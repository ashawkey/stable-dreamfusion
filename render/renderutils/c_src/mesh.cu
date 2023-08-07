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
#include <stdio.h>

#include "common.h"
#include "mesh.h"


//------------------------------------------------------------------------
// Kernels

__global__ void xfmPointsFwdKernel(XfmKernelParams p)
{
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pz = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float mtx[4][4];
    if (threadIdx.x < 16)
        mtx[threadIdx.x % 4][threadIdx.x / 4] = p.matrix.fetch(p.matrix.nhwcIndex(pz, threadIdx.x / 4, threadIdx.x % 4, 0));
    __syncthreads();
    
    if (px >= p.gridSize.x)
        return;

    vec3f pos(
        p.points.fetch(p.points.nhwcIndex(pz, px, 0, 0)),
        p.points.fetch(p.points.nhwcIndex(pz, px, 1, 0)),
        p.points.fetch(p.points.nhwcIndex(pz, px, 2, 0))
    );

    if (p.isPoints)
    {
        p.out.store(p.out.nhwcIndex(pz, px, 0, 0), pos.x * mtx[0][0] + pos.y * mtx[1][0] + pos.z * mtx[2][0] + mtx[3][0]);
        p.out.store(p.out.nhwcIndex(pz, px, 1, 0), pos.x * mtx[0][1] + pos.y * mtx[1][1] + pos.z * mtx[2][1] + mtx[3][1]);
        p.out.store(p.out.nhwcIndex(pz, px, 2, 0), pos.x * mtx[0][2] + pos.y * mtx[1][2] + pos.z * mtx[2][2] + mtx[3][2]);
        p.out.store(p.out.nhwcIndex(pz, px, 3, 0), pos.x * mtx[0][3] + pos.y * mtx[1][3] + pos.z * mtx[2][3] + mtx[3][3]);
    }
    else
    {
        p.out.store(p.out.nhwcIndex(pz, px, 0, 0), pos.x * mtx[0][0] + pos.y * mtx[1][0] + pos.z * mtx[2][0]);
        p.out.store(p.out.nhwcIndex(pz, px, 1, 0), pos.x * mtx[0][1] + pos.y * mtx[1][1] + pos.z * mtx[2][1]);
        p.out.store(p.out.nhwcIndex(pz, px, 2, 0), pos.x * mtx[0][2] + pos.y * mtx[1][2] + pos.z * mtx[2][2]);
    }
}

__global__ void xfmPointsBwdKernel(XfmKernelParams p)
{ 
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pz = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float mtx[4][4];
    if (threadIdx.x < 16)
        mtx[threadIdx.x % 4][threadIdx.x / 4] = p.matrix.fetch(p.matrix.nhwcIndex(pz, threadIdx.x / 4, threadIdx.x % 4, 0));
    __syncthreads();

    if (px >= p.gridSize.x)
        return;

    vec3f pos(
        p.points.fetch(p.points.nhwcIndex(pz, px, 0, 0)),
        p.points.fetch(p.points.nhwcIndex(pz, px, 1, 0)),
        p.points.fetch(p.points.nhwcIndex(pz, px, 2, 0))
    );

    vec4f d_out(
        p.out.fetch(p.out.nhwcIndex(pz, px, 0, 0)),
        p.out.fetch(p.out.nhwcIndex(pz, px, 1, 0)),
        p.out.fetch(p.out.nhwcIndex(pz, px, 2, 0)),
        p.out.fetch(p.out.nhwcIndex(pz, px, 3, 0))
    );

    if (p.isPoints)
    {
        p.points.store_grad(p.points.nhwcIndexContinuous(pz, px, 0, 0), d_out.x * mtx[0][0] + d_out.y * mtx[0][1] + d_out.z * mtx[0][2] + d_out.w * mtx[0][3]);
        p.points.store_grad(p.points.nhwcIndexContinuous(pz, px, 1, 0), d_out.x * mtx[1][0] + d_out.y * mtx[1][1] + d_out.z * mtx[1][2] + d_out.w * mtx[1][3]);
        p.points.store_grad(p.points.nhwcIndexContinuous(pz, px, 2, 0), d_out.x * mtx[2][0] + d_out.y * mtx[2][1] + d_out.z * mtx[2][2] + d_out.w * mtx[2][3]);
    }
    else
    {
        p.points.store_grad(p.points.nhwcIndexContinuous(pz, px, 0, 0), d_out.x * mtx[0][0] + d_out.y * mtx[0][1] + d_out.z * mtx[0][2]);
        p.points.store_grad(p.points.nhwcIndexContinuous(pz, px, 1, 0), d_out.x * mtx[1][0] + d_out.y * mtx[1][1] + d_out.z * mtx[1][2]);
        p.points.store_grad(p.points.nhwcIndexContinuous(pz, px, 2, 0), d_out.x * mtx[2][0] + d_out.y * mtx[2][1] + d_out.z * mtx[2][2]);
    }
}