#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "matmul_kernel.cuh"

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(0));

  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);

  auto c = torch::empty({M, N}, a.options());

  const int BM = 128, BN = 128, BK = 16;
  const int WM = 64, WN = 64;
  const int WNITER = 4;
  const int TM = 8, TN = 4;
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * MM_WARP_SIZE;  // 128

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(NUM_THREADS);

  // Use double-buffer kernel with cp.async for better performance
  matmul_kernel_db<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<grid, block>>>(
    M, N, K, a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

torch::Tensor matmul_cuda_sync(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(0));

  int M = a.size(0);
  int K = a.size(1);
  int N = b.size(1);

  auto c = torch::empty({M, N}, a.options());

  const int BM = 128, BN = 128, BK = 16;
  const int WM = 64, WN = 64;
  const int WNITER = 4;
  const int TM = 8, TN = 4;
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * MM_WARP_SIZE;  // 128

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(NUM_THREADS);

  matmul_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<grid, block>>>(
    M, N, K, a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("matmul", &matmul_cuda);
}