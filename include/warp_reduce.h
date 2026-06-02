#ifndef WARP_REDUCE_H
#define WARP_REDUCE_H

#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__inline__ __device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  return val;
}

#endif // WARP_REDUCE_H