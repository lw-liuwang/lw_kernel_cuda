#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Transpose: naive copy
__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    out[col * rows + row] = in[row * cols + col];
  }
}

// Transpose with shared memory tiling (32x32 tiles to avoid bank conflict)
__global__ void __launch_bounds__(1024)
transpose_smem(const float* in, float* out, int rows, int cols) {
  __shared__ float tile[32][32 + 1];  // padding to avoid bank conflict

  int row = blockIdx.y * 32 + threadIdx.y;
  int col = blockIdx.x * 32 + threadIdx.x;

  if (row < rows && col < cols) {
    tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
  }
  __syncthreads();

  int out_row = blockIdx.x * 32 + threadIdx.y;
  int out_col = blockIdx.y * 32 + threadIdx.x;

  if (out_row < cols && out_col < rows) {
    out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
  }
}

torch::Tensor transpose_cuda(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dtype() == at::kFloat);
  TORCH_CHECK(x.dim() == 2, "transpose expects 2D tensor");

  int rows = x.size(0);
  int cols = x.size(1);
  auto out = torch::empty({cols, rows}, x.options());

  constexpr int TILE = 32;
  dim3 block(TILE, TILE);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

  transpose_smem<<<grid, block>>>(
    x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("transpose", &transpose_cuda);
}