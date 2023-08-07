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
#if defined(__CUDACC__) && defined(BFLOAT16)
#include <cuda_bf16.h> // bfloat16 is float32 compatible with less mantissa bits
#endif

//---------------------------------------------------------------------------------
// CUDA-side Tensor class for in/out parameter parsing. Can be float32 or bfloat16

struct Tensor
{
    void*   val;
    void*   d_val;
    int     dims[4], _dims[4];
    int     strides[4];
    bool    fp16;

#if defined(__CUDA__) && !defined(__CUDA_ARCH__)
    Tensor() : val(nullptr), d_val(nullptr), fp16(true), dims{ 0, 0, 0, 0 }, _dims{ 0, 0, 0, 0 }, strides{ 0, 0, 0, 0 } {}
#endif

#ifdef __CUDACC__
    // Helpers to index and read/write a single element
    __device__ inline int   _nhwcIndex(int n, int h, int w, int c) const { return n * strides[0] + h * strides[1] + w * strides[2] + c * strides[3]; }
    __device__ inline int   nhwcIndex(int n, int h, int w, int c) const { return (dims[0] == 1 ? 0 : n * strides[0]) + (dims[1] == 1 ? 0 : h * strides[1]) + (dims[2] == 1 ? 0 : w * strides[2]) + (dims[3] == 1 ? 0 : c * strides[3]); }
    __device__ inline int   nhwcIndexContinuous(int n, int h, int w, int c) const { return ((n * _dims[1] + h) * _dims[2] + w) * _dims[3] + c; }
#ifdef BFLOAT16
    __device__ inline float fetch(unsigned int idx) const { return fp16 ? __bfloat162float(((__nv_bfloat16*)val)[idx]) : ((float*)val)[idx]; }
    __device__ inline void  store(unsigned int idx, float _val) { if (fp16) ((__nv_bfloat16*)val)[idx] = __float2bfloat16(_val); else ((float*)val)[idx] = _val; }
    __device__ inline void  store_grad(unsigned int idx, float _val) { if (fp16) ((__nv_bfloat16*)d_val)[idx] = __float2bfloat16(_val); else ((float*)d_val)[idx] = _val; }
#else
    __device__ inline float fetch(unsigned int idx) const { return ((float*)val)[idx]; }
    __device__ inline void  store(unsigned int idx, float _val) { ((float*)val)[idx] = _val; }
    __device__ inline void  store_grad(unsigned int idx, float _val) { ((float*)d_val)[idx] = _val; }
#endif

    //////////////////////////////////////////////////////////////////////////////////////////
    // Fetch, use broadcasting for tensor dimensions of size 1
    __device__ inline float fetch1(unsigned int x, unsigned int y, unsigned int z) const
    {
        return fetch(nhwcIndex(z, y, x, 0));
    }

    __device__ inline vec3f fetch3(unsigned int x, unsigned int y, unsigned int z) const
    {
        return vec3f(
            fetch(nhwcIndex(z, y, x, 0)),
            fetch(nhwcIndex(z, y, x, 1)),
            fetch(nhwcIndex(z, y, x, 2))
        );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store, no broadcasting here. Assume we output full res gradient and then reduce using torch.sum outside
    __device__ inline void store(unsigned int x, unsigned int y, unsigned int z, float _val)
    {
        store(_nhwcIndex(z, y, x, 0), _val);
    }

    __device__ inline void store(unsigned int x, unsigned int y, unsigned int z, vec3f _val)
    {
        store(_nhwcIndex(z, y, x, 0), _val.x);
        store(_nhwcIndex(z, y, x, 1), _val.y);
        store(_nhwcIndex(z, y, x, 2), _val.z);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Store gradient , no broadcasting here. Assume we output full res gradient and then reduce using torch.sum outside
    __device__ inline void store_grad(unsigned int x, unsigned int y, unsigned int z, float _val)
    {
        store_grad(nhwcIndexContinuous(z, y, x, 0), _val);
    }

    __device__ inline void store_grad(unsigned int x, unsigned int y, unsigned int z, vec3f _val)
    {
        store_grad(nhwcIndexContinuous(z, y, x, 0), _val.x);
        store_grad(nhwcIndexContinuous(z, y, x, 1), _val.y);
        store_grad(nhwcIndexContinuous(z, y, x, 2), _val.z);
    }
#endif

};
