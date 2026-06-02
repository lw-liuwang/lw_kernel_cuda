#ifndef BLOCK_REDUCE_H
#define BLOCK_REDUCE_H

#include <cuda_runtime.h>

// Block-level sum reduction using warp shuffle + shared memory.
// Supports arbitrary blockDim.x (multiple of 32).
__inline__ __device__ float blockReduceSum(float val) {
  const int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

  __shared__ float warpSums[32];
  if (lane == 0) {
    warpSums[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpSums[tid] : 0.0f;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  } else {
    val = 0.0f;
  }
  return val;
}

// Block-level max reduction using warp shuffle + shared memory.
__inline__ __device__ float blockReduceMax(float val) {
  const int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));

  __shared__ float warpMaxs[32];
  if (lane == 0) {
    warpMaxs[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpMaxs[tid] : -INFINITY;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  } else {
    val = -INFINITY;
  }
  return val;
}

#endif // BLOCK_REDUCE_H