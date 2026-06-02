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

// Reduce kernel: float4 vectorized load + block-level warp-shuffle reduce.
__global__ void reduce_kernel(const float* in, float* out, int n) {
  float sum = 0.0f;

  // float4 vectorized load
  const float4* in4 = reinterpret_cast<const float4*>(in);
  int n4 = n / 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < n4; i += stride) {
    float4 v = in4[i];
    sum += v.x + v.y + v.z + v.w;
  }

  // Tail elements (n % 4 != 0)
  int tail_start = n4 * 4;
  for (int i = tail_start + idx; i < n; i += stride) {
    sum += in[i];
  }

  sum = blockReduceSum(sum);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

torch::Tensor reduce_cuda(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dtype() == at::kFloat);

  int n = x.numel();
  int block_size = 256;
  int grid_size = min((n + block_size - 1) / block_size, 1024);

  // First pass: partial sums
  auto partial = torch::empty({grid_size}, x.options());
  reduce_kernel<<<grid_size, block_size>>>(
    x.data_ptr<float>(), partial.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Second pass: reduce partial sums to single value
  // FIX: use block_size = min(grid_size, 256) instead of grid_size directly
  // grid_size may be > 1024, which is too large for blockDim
  int grid_size2 = 1;
  int block_size2 = min(grid_size, 256);
  auto out = torch::empty({1}, x.options());
  reduce_kernel<<<grid_size2, block_size2>>>(
    partial.data_ptr<float>(), out.data_ptr<float>(), grid_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("reduce", &reduce_cuda);
}