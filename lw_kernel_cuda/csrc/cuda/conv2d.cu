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

// Conv2D via im2col + GEMM (using matmul_kernel from shared header)

__global__ void im2col_kernel(const float* x, float* cols,
                               int N, int C, int H, int W,
                               int K, int S, int P,
                               int OH, int OW) {
  int num_cols = C * K * K;
  int total = N * OH * OW * num_cols;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int row = idx / num_cols;   // which row in cols = n*OH*OW + oh*OW + ow
  int col = idx % num_cols;   // which col in cols = c*K*K + kk

  int ow = row % OW;
  int oh = (row / OW) % OH;
  int n = row / (OH * OW);

  int kk = col % (K * K);
  int c = col / (K * K);
  int ky = kk / K;
  int kx = kk % K;

  int h = oh * S - P + ky;
  int w = ow * S - P + kx;

  int src_idx = n * (C * H * W) + c * (H * W) + h * W + w;
  cols[idx] = (h >= 0 && h < H && w >= 0 && w < W) ? x[src_idx] : 0.0f;
}

torch::Tensor conv2d_cuda(torch::Tensor x, torch::Tensor weight,
                           torch::Tensor bias, int64_t stride, int64_t padding) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  TORCH_CHECK(x.dtype() == at::kFloat);
  TORCH_CHECK(weight.dtype() == at::kFloat);
  TORCH_CHECK(x.dim() == 4, "x expects (N, C, H, W)");
  TORCH_CHECK(weight.dim() == 4, "weight expects (OC, IC, KH, KW)");

  int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
  int OC = weight.size(0), IC = weight.size(1);
  int K = weight.size(2);  // kernel size (square)
  TORCH_CHECK(IC == C, "in_channels mismatch");

  int OH = (H + 2 * padding - K) / stride + 1;
  int OW = (W + 2 * padding - K) / stride + 1;

  // im2col: output is (N * OH * OW, C * K * K)
  int col_rows = N * OH * OW;
  int col_cols = C * K * K;
  auto cols = torch::empty({col_rows, col_cols}, x.options());

  // Launch im2col kernel
  int total_im2col = N * C * K * K * OH * OW;
  int block_size = 256;
  int grid_size = (total_im2col + block_size - 1) / block_size;
  im2col_kernel<<<grid_size, block_size>>>(
    x.data_ptr<float>(), cols.data_ptr<float>(),
    N, C, H, W, K, stride, padding, OH, OW);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // GEMM: cols (N*OH*OW, C*K*K), weight (OC, C*K*K)
  // We need out = cols @ weight.T  →  (N*OH*OW, OC)
  auto weight_rect = weight.view({OC, col_cols});
  auto weight_t = weight_rect.transpose(0, 1).contiguous();  // [C*K*K, OC]

  int M = col_rows;
  int Kdim = col_cols;
  int Ndim = OC;
  auto out_2d = torch::empty({M, Ndim}, x.options());

  // Use the same matmul_kernel from the shared header
  const int BM = 128, BN = 128, BK = 16;
  const int WM = 64, WN = 64;
  const int WNITER = 4;
  const int TM = 8, TN = 4;
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * MM_WARP_SIZE;  // 128

  dim3 grid_gemm((Ndim + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block_gemm(NUM_THREADS);

  matmul_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<grid_gemm, block_gemm>>>(
    M, Ndim, Kdim,
    cols.data_ptr<float>(),
    weight_t.data_ptr<float>(),
    out_2d.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Reshape to (N, OH, OW, OC) then permute to (N, OC, OH, OW)
  auto out_4d = out_2d.view({N, OH, OW, OC}).permute({0, 3, 1, 2}).contiguous();

  // Add bias
  if (bias.defined() && bias.numel() > 0) {
    out_4d = out_4d + bias.view({1, OC, 1, 1});
  }

  return out_4d;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("conv2d", &conv2d_cuda);
}