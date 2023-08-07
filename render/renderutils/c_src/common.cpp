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

#include <cuda_runtime.h>
#include <algorithm>

//------------------------------------------------------------------------
// Block and grid size calculators for kernel launches.

dim3 getLaunchBlockSize(int maxWidth, int maxHeight, dim3 dims)
{
    int maxThreads = maxWidth * maxHeight;
    if (maxThreads <= 1 || (dims.x * dims.y) <= 1)
        return dim3(1, 1, 1); // Degenerate.

    // Start from max size.
    int bw = maxWidth;
    int bh = maxHeight;

    // Optimizations for weirdly sized buffers.
    if (dims.x < bw)
    {
        // Decrease block width to smallest power of two that covers the buffer width.
        while ((bw >> 1) >= dims.x)
            bw >>= 1;

        // Maximize height.
        bh = maxThreads / bw;
        if (bh > dims.y)
            bh = dims.y;
    }
    else if (dims.y < bh)
    {
        // Halve height and double width until fits completely inside buffer vertically.
        while (bh > dims.y)
        {
            bh >>= 1;
            if (bw < dims.x)
                bw <<= 1;
        }
    }

    // Done.
    return dim3(bw, bh, 1);
}

// returns the size of a block that can be reduced using horizontal SIMD operations (e.g. __shfl_xor_sync)
dim3 getWarpSize(dim3 blockSize)
{
    return dim3(
        std::min(blockSize.x, 32u), 
        std::min(std::max(32u / blockSize.x, 1u), std::min(32u, blockSize.y)), 
        std::min(std::max(32u / (blockSize.x * blockSize.y), 1u), std::min(32u, blockSize.z))
    );
}

dim3 getLaunchGridSize(dim3 blockSize, dim3 dims)
{
    dim3 gridSize;
    gridSize.x = (dims.x  - 1) / blockSize.x + 1;
    gridSize.y = (dims.y - 1) / blockSize.y + 1;
    gridSize.z = (dims.z  - 1) / blockSize.z + 1;
    return gridSize;
}

//------------------------------------------------------------------------
