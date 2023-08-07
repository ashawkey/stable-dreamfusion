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

struct vec3f
{
    float x, y, z;

#ifdef __CUDACC__
    __device__ vec3f() { }
    __device__ vec3f(float v) { x = v; y = v; z = v; }
    __device__ vec3f(float _x, float _y, float _z) { x = _x; y = _y; z = _z; }
    __device__ vec3f(float3 v) { x = v.x; y = v.y; z = v.z; }

    __device__ inline vec3f& operator+=(const vec3f& b) { x += b.x; y += b.y; z += b.z; return *this; }
    __device__ inline vec3f& operator-=(const vec3f& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    __device__ inline vec3f& operator*=(const vec3f& b) { x *= b.x; y *= b.y; z *= b.z; return *this; }
    __device__ inline vec3f& operator/=(const vec3f& b) { x /= b.x; y /= b.y; z /= b.z; return *this; }
#endif
};

#ifdef __CUDACC__
__device__ static inline vec3f operator+(const vec3f& a, const vec3f& b) { return vec3f(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ static inline vec3f operator-(const vec3f& a, const vec3f& b) { return vec3f(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ static inline vec3f operator*(const vec3f& a, const vec3f& b) { return vec3f(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ static inline vec3f operator/(const vec3f& a, const vec3f& b) { return vec3f(a.x / b.x, a.y / b.y, a.z / b.z); }
__device__ static inline vec3f operator-(const vec3f& a) { return vec3f(-a.x, -a.y, -a.z); }

__device__ static inline float sum(vec3f a)
{
    return a.x + a.y + a.z;
}

__device__ static inline vec3f cross(vec3f a, vec3f b)
{
    vec3f out;
    out.x = a.y * b.z - a.z * b.y;
    out.y = a.z * b.x - a.x * b.z;
    out.z = a.x * b.y - a.y * b.x;
    return out;
}

__device__ static inline void bwdCross(vec3f a, vec3f b, vec3f &d_a, vec3f &d_b, vec3f d_out)
{
    d_a.x += d_out.z * b.y - d_out.y * b.z;
    d_a.y += d_out.x * b.z - d_out.z * b.x;
    d_a.z += d_out.y * b.x - d_out.x * b.y;

    d_b.x += d_out.y * a.z - d_out.z * a.y;
    d_b.y += d_out.z * a.x - d_out.x * a.z;
    d_b.z += d_out.x * a.y - d_out.y * a.x;
}

__device__ static inline float dot(vec3f a, vec3f b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static inline void bwdDot(vec3f a, vec3f b, vec3f& d_a, vec3f& d_b, float d_out)
{
    d_a.x += d_out * b.x; d_a.y += d_out * b.y; d_a.z += d_out * b.z;
    d_b.x += d_out * a.x; d_b.y += d_out * a.y; d_b.z += d_out * a.z;
}

__device__ static inline vec3f reflect(vec3f x, vec3f n)
{
    return n * 2.0f * dot(n, x) - x;
}

__device__ static inline void bwdReflect(vec3f x, vec3f n, vec3f& d_x, vec3f& d_n, const vec3f d_out)
{
    d_x.x += d_out.x * (2 * n.x * n.x - 1) + d_out.y * (2 * n.x * n.y) + d_out.z * (2 * n.x * n.z);
    d_x.y += d_out.x * (2 * n.x * n.y) + d_out.y * (2 * n.y * n.y - 1) + d_out.z * (2 * n.y * n.z);
    d_x.z += d_out.x * (2 * n.x * n.z) + d_out.y * (2 * n.y * n.z) + d_out.z * (2 * n.z * n.z - 1);

    d_n.x += d_out.x * (2 * (2 * n.x * x.x + n.y * x.y + n.z * x.z)) + d_out.y * (2 * n.y * x.x) + d_out.z * (2 * n.z * x.x);
    d_n.y += d_out.x * (2 * n.x * x.y) + d_out.y * (2 * (n.x * x.x + 2 * n.y * x.y + n.z * x.z)) + d_out.z * (2 * n.z * x.y);
    d_n.z += d_out.x * (2 * n.x * x.z) + d_out.y * (2 * n.y * x.z) + d_out.z * (2 * (n.x * x.x + n.y * x.y + 2 * n.z * x.z));
}

__device__ static inline vec3f safeNormalize(vec3f v)
{
    float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return l > 0.0f ? (v / l) : vec3f(0.0f);
}

__device__ static inline void bwdSafeNormalize(const vec3f v, vec3f& d_v, const vec3f d_out)
{

    float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (l > 0.0f)
    {
        float fac = 1.0 / powf(v.x * v.x + v.y * v.y + v.z * v.z, 1.5f);
        d_v.x += (d_out.x * (v.y * v.y + v.z * v.z) - d_out.y * (v.x * v.y) - d_out.z * (v.x * v.z)) * fac;
        d_v.y += (d_out.y * (v.x * v.x + v.z * v.z) - d_out.x * (v.y * v.x) - d_out.z * (v.y * v.z)) * fac;
        d_v.z += (d_out.z * (v.x * v.x + v.y * v.y) - d_out.x * (v.z * v.x) - d_out.y * (v.z * v.y)) * fac;
    }
}

#endif