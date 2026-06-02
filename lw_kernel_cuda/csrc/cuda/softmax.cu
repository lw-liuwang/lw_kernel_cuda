#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "block_reduce.h"

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

/**
 * Softmax kernel — optimized version.
 * Key optimizations:
 * 1. Fused expf + sum: compute expf in registers and accumulate sum in same loop
 *    (eliminates global memory roundtrip that old kernel had)
 * 2. float4 vectorized load/store for global memory
 * 3. blockReduceMax/Sum from shared header (warp shuffle + smem)
 */
__global__ void softmax_kernel(const float* __restrict__ inp,
                                float* __restrict__ out,
                                int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;

  const float* x = inp + row * cols;
  float* y = out + row * cols;
  int tid = threadIdx.x;

  // Check if this row's starting address is 16-byte aligned for float4 access
  // When cols % 4 != 0, rows beyond row 0 may start at non-16-byte-aligned addresses
  bool aligned = (((uintptr_t)x) & 15) == 0;
  int cols4 = aligned ? (cols / 4) : 0;
  int tail = cols4 * 4;

  // ----- Step 1: max reduction (float4 vectorized when aligned) -----
  float maxval = -INFINITY;
  if (aligned && cols4 > 0) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid; i < cols4; i += blockDim.x) {
      float4 v = x4[i];
      maxval = fmaxf(maxval, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
  }
  for (int i = tail + tid; i < cols; i += blockDim.x) {
    maxval = fmaxf(maxval, x[i]);
  }
  maxval = blockReduceMax(maxval);
  // Broadcast max from thread 0 to all threads (blockReduceMax only gives correct result to warp 0)
  __shared__ float s_max;
  if (threadIdx.x == 0) s_max = maxval;
  __syncthreads();
  maxval = s_max;

  // ----- Step 2: fused expf + sum (float4 loads when aligned, NO intermediate global write/read) -----
  float sumval = 0.0f;
  if (aligned && cols4 > 0) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    for (int i = tid; i < cols4; i += blockDim.x) {
      float4 v = x4[i];
      float ex = expf(v.x - maxval);
      float ey = expf(v.y - maxval);
      float ez = expf(v.z - maxval);
      float ew = expf(v.w - maxval);
      y4[i] = make_float4(ex, ey, ez, ew);
      sumval += ex + ey + ez + ew;
    }
  }
  for (int i = tail + tid; i < cols; i += blockDim.x) {
    float e = expf(x[i] - maxval);
    y[i] = e;
    sumval += e;
  }
  sumval = blockReduceSum(sumval);
  // Broadcast sum from thread 0 to all threads
  __shared__ float s_sum;
  if (threadIdx.x == 0) s_sum = sumval;
  __syncthreads();
  sumval = s_sum;

  // ----- Step 3: normalize (float4 vectorized store when aligned) -----
  float inv_sum = 1.0f / sumval;
  if (aligned && cols4 > 0) {
    float4* y4 = reinterpret_cast<float4*>(y);
    for (int i = tid; i < cols4; i += blockDim.x) {
      float4 e = y4[i];
      y4[i] = make_float4(e.x * inv_sum, e.y * inv_sum, e.z * inv_sum, e.w * inv_sum);
    }
  }
  for (int i = tail + tid; i < cols; i += blockDim.x) {
    y[i] *= inv_sum;
  }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dtype() == at::kFloat);
  TORCH_CHECK(x.dim() == 2, "softmax expects 2D tensor (rows, cols)");

  int rows = x.size(0);
  int cols = x.size(1);
  auto out = torch::empty_like(x);

  int block_size = 256;
  if (cols < 256) block_size = 32;
  dim3 grid(rows);
  dim3 block(block_size);

  softmax_kernel<<<grid, block>>>(
    x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("softmax", &softmax_cuda);
}