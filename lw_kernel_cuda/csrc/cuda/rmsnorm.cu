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

// RMSNorm: one block per row, float4 vectorized load, block reduce sum.
__global__ void __launch_bounds__(256)
rmsnorm_kernel(const float* in, const float* weight,
                                float* out, int rows, int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows) return;

  const float* row_in = in + row * cols;
  float* row_out = out + row * cols;
  int tid = threadIdx.x;
  int n = cols;

  // Compute sum of squares with float4 vectorization
  float sum = 0.0f;
  int vec_size = 4;
  int vec_num = n / vec_size;
  int vec_remain = n % vec_size;

  const float4* in_vec = reinterpret_cast<const float4*>(row_in);
  for (int i = tid; i < vec_num; i += blockDim.x) {
    float4 v = in_vec[i];
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  // Tail elements
  int tail_off = vec_num * vec_size;
  for (int i = tail_off + tid; i < n; i += blockDim.x) {
    sum += row_in[i] * row_in[i];
  }

  sum = blockReduceSum(sum);

  __shared__ float shared_rms;
  if (tid == 0) {
    shared_rms = rsqrtf(sum / (float)n + eps);
  }
  __syncthreads();

  float rms = shared_rms;

  // Apply weight and normalize using float4
  const float4* w_vec = reinterpret_cast<const float4*>(weight);
  float4* out_vec = reinterpret_cast<float4*>(row_out);
  for (int i = tid; i < vec_num; i += blockDim.x) {
    float4 iv = in_vec[i];
    float4 wv = w_vec[i];
    out_vec[i] = make_float4(iv.x * wv.x * rms, iv.y * wv.y * rms,
                              iv.z * wv.z * rms, iv.w * wv.w * rms);
  }
  for (int i = tail_off + tid; i < n; i += blockDim.x) {
    row_out[i] = row_in[i] * weight[i] * rms;
  }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  TORCH_CHECK(x.dtype() == at::kFloat);
  TORCH_CHECK(weight.dtype() == at::kFloat);
  TORCH_CHECK(x.dim() == 2, "rmsnorm expects 2D tensor (rows, cols)");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(x.size(1) == weight.size(0));

  int rows = x.size(0);
  int cols = x.size(1);
  auto out = torch::empty_like(x);

  int block_size = 256;
  dim3 grid(rows);
  dim3 block(block_size);

  rmsnorm_kernel<<<grid, block>>>(
    x.data_ptr<float>(), weight.data_ptr<float>(),
    out.data_ptr<float>(), rows, cols, eps);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("rmsnorm", &rmsnorm_cuda);
}