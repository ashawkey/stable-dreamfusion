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
#include "cubemap.h"
#include <float.h>

// https://cgvr.cs.uni-bremen.de/teaching/cg_literatur/Spherical,%20Cubic,%20and%20Parabolic%20Environment%20Mappings.pdf
__device__ float pixel_area(int x, int y, int N)
{
    if (N > 1)
    {
        int H = N / 2;
        x = abs(x - H);
        y = abs(y - H);
        float dx = atan((float)(x + 1) / (float)H) - atan((float)x / (float)H);
        float dy = atan((float)(y + 1) / (float)H) - atan((float)y / (float)H);
        return dx * dy;
    }
    else
        return 1;
}

__device__ vec3f cube_to_dir(int x, int y, int side, int N)
{
    float fx = 2.0f * (((float)x + 0.5f) / (float)N) - 1.0f;
    float fy = 2.0f * (((float)y + 0.5f) / (float)N) - 1.0f;
    switch (side)
    {
        case 0: return safeNormalize(vec3f(1, -fy, -fx));
        case 1: return safeNormalize(vec3f(-1, -fy, fx));
        case 2: return safeNormalize(vec3f(fx, 1, fy));
        case 3: return safeNormalize(vec3f(fx, -1, -fy));
        case 4: return safeNormalize(vec3f(fx, -fy, 1));
        case 5: return safeNormalize(vec3f(-fx, -fy, -1));
    }
    return vec3f(0,0,0); // Unreachable
}

__device__ vec3f dir_to_side(int side, vec3f v)
{
    switch (side)
    {
    case 0: return vec3f(-v.z, -v.y,  v.x);
    case 1: return vec3f( v.z, -v.y, -v.x);
    case 2: return vec3f( v.x,  v.z,  v.y);
    case 3: return vec3f( v.x, -v.z, -v.y);
    case 4: return vec3f( v.x, -v.y,  v.z);
    case 5: return vec3f(-v.x, -v.y, -v.z);
    }
    return vec3f(0,0,0); // Unreachable
}

__device__ void extents_1d(float x, float z, float theta, float& _min, float& _max)
{
    float l = sqrtf(x * x + z * z);
    float pxr = x + z * tan(theta) * l, pzr = z - x * tan(theta) * l;
    float pxl = x - z * tan(theta) * l, pzl = z + x * tan(theta) * l;
    if (pzl <= 0.00001f)
        _min = pxl > 0.0f ? FLT_MAX : -FLT_MAX;
    else
        _min = pxl / pzl;
    if (pzr <= 0.00001f)
        _max = pxr > 0.0f ? FLT_MAX : -FLT_MAX;
    else
        _max = pxr / pzr;
}

__device__ void dir_extents(int side, int N, vec3f v, float theta, int &_xmin, int& _xmax, int& _ymin, int& _ymax)
{
    vec3f c = dir_to_side(side, v); // remap to (x,y,z) where side is at z = 1

    if (theta < 0.785398f) // PI/4
    {
        float xmin, xmax, ymin, ymax;
        extents_1d(c.x, c.z, theta, xmin, xmax);
        extents_1d(c.y, c.z, theta, ymin, ymax);

        if (xmin > 1.0f || xmax < -1.0f || ymin > 1.0f || ymax < -1.0f)
        {
            _xmin = -1; _xmax = -1; _ymin = -1; _ymax = -1; // Bad aabb
        }
        else
        {
            _xmin = (int)min(max((xmin + 1.0f) * (0.5f * (float)N), 0.0f), (float)(N - 1));
            _xmax = (int)min(max((xmax + 1.0f) * (0.5f * (float)N), 0.0f), (float)(N - 1));
            _ymin = (int)min(max((ymin + 1.0f) * (0.5f * (float)N), 0.0f), (float)(N - 1));
            _ymax = (int)min(max((ymax + 1.0f) * (0.5f * (float)N), 0.0f), (float)(N - 1));
        }
    }
    else
    {
            _xmin = 0.0f;
            _xmax = (float)(N-1);
            _ymin = 0.0f;
            _ymax = (float)(N-1);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Diffuse kernel
__global__ void DiffuseCubemapFwdKernel(DiffuseCubemapKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    int Npx = p.cubemap.dims[1];
    vec3f N = cube_to_dir(px, py, pz, Npx);

    vec3f col(0);

    for (int s = 0; s < p.cubemap.dims[0]; ++s)
    {
        for (int y = 0; y < Npx; ++y)
        {
            for (int x = 0; x < Npx; ++x)
            {
                vec3f L = cube_to_dir(x, y, s, Npx);
                float costheta = min(max(dot(N, L), 0.0f), 0.999f);
                float w = costheta * pixel_area(x, y, Npx) / 3.141592f; // pi = area of positive hemisphere
                col += p.cubemap.fetch3(x, y, s) * w;
            }
        }
    }

    p.out.store(px, py, pz, col);
}

__global__ void DiffuseCubemapBwdKernel(DiffuseCubemapKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    int Npx = p.cubemap.dims[1];
    vec3f N = cube_to_dir(px, py, pz, Npx);
    vec3f grad = p.out.fetch3(px, py, pz);

    for (int s = 0; s < p.cubemap.dims[0]; ++s)
    {
        for (int y = 0; y < Npx; ++y)
        {
            for (int x = 0; x < Npx; ++x)
            {
                vec3f L = cube_to_dir(x, y, s, Npx);
                float costheta = min(max(dot(N, L), 0.0f), 0.999f);
                float w = costheta * pixel_area(x, y, Npx) / 3.141592f; // pi = area of positive hemisphere
                atomicAdd((float*)p.cubemap.d_val + p.cubemap.nhwcIndexContinuous(s, y, x, 0), grad.x * w);
                atomicAdd((float*)p.cubemap.d_val + p.cubemap.nhwcIndexContinuous(s, y, x, 1), grad.y * w);
                atomicAdd((float*)p.cubemap.d_val + p.cubemap.nhwcIndexContinuous(s, y, x, 2), grad.z * w);
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// GGX splitsum kernel 

__device__ inline float ndfGGX(const float alphaSqr, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, 0.0, 1.0f);
    float d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1.0f;
    return alphaSqr / (d * d * M_PI);
}

__global__ void SpecularBoundsKernel(SpecularBoundsKernelParams p)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    int Npx = p.gridSize.x;
    vec3f VNR = cube_to_dir(px, py, pz, Npx);

    const int TILE_SIZE = 16;

    // Brute force entire cubemap and compute bounds for the cone
    for (int s = 0; s < p.gridSize.z; ++s)
    {
        // Assume empty BBox 
        int _min_x = p.gridSize.x - 1, _max_x = 0;
        int _min_y = p.gridSize.y - 1, _max_y = 0;
        
        // For each (8x8) tile
        for (int tx = 0; tx < (p.gridSize.x + TILE_SIZE - 1) / TILE_SIZE; tx++)
        {
            for (int ty = 0; ty < (p.gridSize.y + TILE_SIZE - 1) / TILE_SIZE; ty++)
            {
                // Compute tile extents
                int tsx = tx * TILE_SIZE, tsy = ty * TILE_SIZE;
                int tex = min((tx + 1) * TILE_SIZE, p.gridSize.x), tey = min((ty + 1) * TILE_SIZE, p.gridSize.y);

                // Use some blunt interval arithmetics to cull tiles
                vec3f L0 = cube_to_dir(tsx, tsy, s, Npx), L1 = cube_to_dir(tex, tsy, s, Npx);
                vec3f L2 = cube_to_dir(tsx, tey, s, Npx), L3 = cube_to_dir(tex, tey, s, Npx);
                
                float minx = min(min(L0.x, L1.x), min(L2.x, L3.x)), maxx = max(max(L0.x, L1.x), max(L2.x, L3.x));
                float miny = min(min(L0.y, L1.y), min(L2.y, L3.y)), maxy = max(max(L0.y, L1.y), max(L2.y, L3.y));
                float minz = min(min(L0.z, L1.z), min(L2.z, L3.z)), maxz = max(max(L0.z, L1.z), max(L2.z, L3.z));

                float maxdp = max(minx * VNR.x, maxx * VNR.x) + max(miny * VNR.y, maxy * VNR.y) + max(minz * VNR.z, maxz * VNR.z);
                if (maxdp >= p.costheta_cutoff)
                {
                    // Test all pixels in tile.
                    for (int y = tsy; y < tey; ++y)
                    {
                        for (int x = tsx; x < tex; ++x)
                        {
                            vec3f L = cube_to_dir(x, y, s, Npx);
                            if (dot(L, VNR) >= p.costheta_cutoff)
                            {
                                _min_x = min(_min_x, x);
                                _max_x = max(_max_x, x);
                                _min_y = min(_min_y, y);
                                _max_y = max(_max_y, y);
                            }
                        }
                    }
                }
            }
        }
        p.out.store(p.out._nhwcIndex(pz, py, px, s * 4 + 0), _min_x);
        p.out.store(p.out._nhwcIndex(pz, py, px, s * 4 + 1), _max_x);
        p.out.store(p.out._nhwcIndex(pz, py, px, s * 4 + 2), _min_y);
        p.out.store(p.out._nhwcIndex(pz, py, px, s * 4 + 3), _max_y);
    }
}

__global__ void SpecularCubemapFwdKernel(SpecularCubemapKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    int Npx = p.cubemap.dims[1];
    vec3f VNR = cube_to_dir(px, py, pz, Npx);

    float alpha = p.roughness * p.roughness;
    float alphaSqr = alpha * alpha;

    float wsum = 0.0f;
    vec3f col(0);
    for (int s = 0; s < p.cubemap.dims[0]; ++s)
    {
        int xmin, xmax, ymin, ymax;
        xmin = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 0));
        xmax = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 1));
        ymin = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 2));
        ymax = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 3));

        if (xmin <= xmax)
        {
            for (int y = ymin; y <= ymax; ++y)
            {
                for (int x = xmin; x <= xmax; ++x)
                {
                    vec3f L = cube_to_dir(x, y, s, Npx);
                    if (dot(L, VNR) >= p.costheta_cutoff)
                    {
                        vec3f H = safeNormalize(L + VNR);

                        float wiDotN = max(dot(L, VNR), 0.0f);
                        float VNRDotH = max(dot(VNR, H), 0.0f);

                        float w = wiDotN * ndfGGX(alphaSqr, VNRDotH) * pixel_area(x, y, Npx) / 4.0f;
                        col += p.cubemap.fetch3(x, y, s) * w;
                        wsum += w;
                    }
                }
            }
        }
    }

    p.out.store(p.out._nhwcIndex(pz, py, px, 0), col.x);
    p.out.store(p.out._nhwcIndex(pz, py, px, 1), col.y);
    p.out.store(p.out._nhwcIndex(pz, py, px, 2), col.z);
    p.out.store(p.out._nhwcIndex(pz, py, px, 3), wsum);
}

__global__ void SpecularCubemapBwdKernel(SpecularCubemapKernelParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.gridSize.x || py >= p.gridSize.y || pz >= p.gridSize.z)
        return;

    int Npx = p.cubemap.dims[1];
    vec3f VNR = cube_to_dir(px, py, pz, Npx);

    vec3f grad = p.out.fetch3(px, py, pz);

    float alpha = p.roughness * p.roughness;
    float alphaSqr = alpha * alpha;

    vec3f col(0);
    for (int s = 0; s < p.cubemap.dims[0]; ++s)
    {
        int xmin, xmax, ymin, ymax;
        xmin = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 0));
        xmax = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 1));
        ymin = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 2));
        ymax = (int)p.bounds.fetch(p.bounds._nhwcIndex(pz, py, px, s * 4 + 3));

        if (xmin <= xmax)
        {
            for (int y = ymin; y <= ymax; ++y)
            {
                for (int x = xmin; x <= xmax; ++x)
                {
                    vec3f L = cube_to_dir(x, y, s, Npx);
                    if (dot(L, VNR) >= p.costheta_cutoff)
                    {
                        vec3f H = safeNormalize(L + VNR);

                        float wiDotN = max(dot(L, VNR), 0.0f);
                        float VNRDotH = max(dot(VNR, H), 0.0f);

                        float w = wiDotN * ndfGGX(alphaSqr, VNRDotH) * pixel_area(x, y, Npx) / 4.0f;

                        atomicAdd((float*)p.cubemap.d_val + p.cubemap.nhwcIndexContinuous(s, y, x, 0), grad.x * w);
                        atomicAdd((float*)p.cubemap.d_val + p.cubemap.nhwcIndexContinuous(s, y, x, 1), grad.y * w);
                        atomicAdd((float*)p.cubemap.d_val + p.cubemap.nhwcIndexContinuous(s, y, x, 2), grad.z * w);
                    }
                }
            }
        }
    }
}
