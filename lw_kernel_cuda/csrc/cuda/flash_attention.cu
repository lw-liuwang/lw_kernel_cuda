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
 * FlashAttention — optimized with larger tiles and blockReduce.
 *
 * Architecture:
 *   - blockDim = 128 threads: each thread handles 1 element of d=128
 *   - One block per (head, query_row): grid = (B*H, seqlen)
 *   - Br=1, Bc=32: processes one query row at a time, K/V in tiles of 32
 *   - blockReduceSum for the dot product Q·K across all 128 threads
 *   - Online softmax update with per-thread state
 *
 * Shared memory: sK(Bc x d) + sV(Bc x d) = 2 * 32 * 128 * 4 = 32 KB
 */
template <int Bc>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen, int stride_head, int d, float sm_scale) {

  int tid = threadIdx.x;
  int q_row = blockIdx.y;
  if (q_row >= seqlen) return;

  int head_base = blockIdx.x * stride_head;
  Q += head_base; K += head_base; V += head_base; O += head_base;

  // Each thread handles 1 element in the d dimension
  // Map tid=0..127 to d=0..127
  if (tid >= d) return;

  // Load Q element
  float q_val = Q[q_row * d + tid];

  // Online softmax state (per-thread)
  float m = -INFINITY;
  float s = 0.0f;
  float acc = 0.0f;

  int num_tiles = (seqlen + Bc - 1) / Bc;

  for (int tile = 0; tile < num_tiles; tile++) {
    // Shared memory for K and V tiles
    __shared__ float sK[Bc * 128];  // Bc x d (d=128 max)
    __shared__ float sV[Bc * 128];

    // Load K and V tile into shared memory
    // Each thread loads elements using grid-stride loop
    for (int i = tid; i < Bc * d; i += blockDim.x) {
      int bc_idx = i / d;
      int di = i % d;
      int k_pos = tile * Bc + bc_idx;
      if (k_pos < seqlen) {
        sK[bc_idx * d + di] = K[k_pos * d + di];
        sV[bc_idx * d + di] = V[k_pos * d + di];
      } else {
        sK[bc_idx * d + di] = 0.0f;
        sV[bc_idx * d + di] = 0.0f;
      }
    }
    __syncthreads();

    // Process each position in this tile
    for (int p = 0; p < Bc; p++) {
      int k_pos = tile * Bc + p;
      if (k_pos >= seqlen) break;

      // Dot product: Q[q_row] · K[k_pos] via blockReduceSum
      float partial = q_val * sK[p * d + tid];

      // blockReduceSum: reduces across all 128 threads
      // thread 0 gets the total sum
      float total = blockReduceSum(partial);

      // Broadcast total to all threads via shared memory
      __shared__ float s_total;
      if (tid == 0) s_total = total;
      __syncthreads();
      float score = s_total * sm_scale;

      // Online softmax update
      float new_m = fmaxf(m, score);
      float rescale = expf(m - new_m);
      float p_val = expf(score - new_m);
      s = s * rescale + p_val;
      m = new_m;

      // Update output accumulator
      acc = acc * rescale + p_val * sV[p * d + tid];
    }
    __syncthreads();
  }

  // Normalize and write output
  O[q_row * d + tid] = acc / s;
}

torch::Tensor flash_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
  TORCH_CHECK(q.dtype() == at::kFloat);
  TORCH_CHECK(k.dtype() == at::kFloat);
  TORCH_CHECK(v.dtype() == at::kFloat);
  TORCH_CHECK(q.dim() == 4, "q expects (B, H, seqlen, dim)");
  TORCH_CHECK(k.sizes() == q.sizes());
  TORCH_CHECK(v.sizes() == q.sizes());

  int bs = q.size(0), head = q.size(1);
  int seqlen = q.size(2), dim = q.size(3);

  TORCH_CHECK(dim <= 128, "flash_attention supports dim<=128, got ", dim);
  TORCH_CHECK(dim >= 32, "flash_attention requires dim>=32");

  float sm_scale = 1.0f / sqrtf(static_cast<float>(dim));
  int stride_head = seqlen * dim;

  auto out = torch::zeros_like(q);

  const int Bc = 32;
  int Gc = bs * head;
  int Gr = seqlen;

  dim3 grid(Gc, Gr);
  int block_size = dim < 32 ? 32 : dim;  // each thread handles 1 element, min 32 for occupancy
  dim3 block(block_size);

  flash_attention_kernel<Bc><<<grid, block>>>(
    q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
    out.data_ptr<float>(), seqlen, stride_head, dim, sm_scale);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("flash_attention", &flash_attention_cuda);
}