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
 * Warp-level reduce sum: reduces across all 32 threads in a warp.
 * Result is available to ALL threads in the warp (not just lane 0).
 * ~15 cycles, no shared memory or __syncthreads() needed.
 */
__device__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

// ----------------------------------------------------------------
// Br=1 fallback kernel (for d=64): one query row per block
// ----------------------------------------------------------------
template <int Bc>
__global__ void flash_attention_br1_kernel(
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

  if (tid >= d) return;

  float q_val = Q[q_row * d + tid];

  float m = -INFINITY;
  float s = 0.0f;
  float acc = 0.0f;

  int num_tiles = (seqlen + Bc - 1) / Bc;

  for (int tile = 0; tile < num_tiles; tile++) {
    __shared__ float sK[Bc * 128];
    __shared__ float sV[Bc * 128];

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

    for (int p = 0; p < Bc; p++) {
      int k_pos = tile * Bc + p;
      if (k_pos >= seqlen) break;

      float partial = q_val * sK[p * d + tid];
      float total = blockReduceSum(partial);

      __shared__ float s_total;
      if (tid == 0) s_total = total;
      __syncthreads();
      float score = s_total * sm_scale;

      float new_m = fmaxf(m, score);
      float rescale = expf(m - new_m);
      float p_val = expf(score - new_m);
      s = s * rescale + p_val;
      m = new_m;

      acc = acc * rescale + p_val * sV[p * d + tid];
    }
    __syncthreads();
  }

  O[q_row * d + tid] = acc / s;
}

// ----------------------------------------------------------------
// Br=4 warp-reduce kernel (for d=128): 4 query rows per block
// ----------------------------------------------------------------
/**
 * Architecture:
 *   - blockDim = 128 (4 warps x 32 threads)
 *   - Each warp handles 1 query row → Br = 4
 *   - grid = (B*H, seqlen/Br)
 *   - Bc = 32: K/V tile of 32 positions
 *   - Each thread handles d/warpSize = 4 elements via float4
 *   - Dot product: per-thread float4 dot → warpReduceSum (pure shuffle, no smem)
 *   - Online softmax per-warp state
 */
template <int Bc>
__global__ void flash_attention_br4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen, int stride_head, int d, float sm_scale) {

  int tid = threadIdx.x;
  int warp = tid >> 5;       // warp = 0..3, each handles one query row
  int lane = tid & 31;       // lane = 0..31, each handles 4 d-elements (d=128)

  int head_base = blockIdx.x * stride_head;
  int q_base_row = blockIdx.y * 4;

  Q += head_base; K += head_base; V += head_base; O += head_base;

  int q_row = q_base_row + warp;

  // Each thread handles 4 elements in the d dimension (d=128, 128/32=4)

  // Load Q values (float4)
  float4 q_val;
  if (q_row < seqlen) {
    q_val = reinterpret_cast<const float4*>(Q)[q_row * (d / 4) + lane];
  } else {
    q_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  // Online softmax state (per-thread, shared across warp via shuffle broadcast)
  float m = -INFINITY;
  float s = 0.0f;
  float acc_x = 0.0f, acc_y = 0.0f, acc_z = 0.0f, acc_w = 0.0f;

  int num_tiles = (seqlen + Bc - 1) / Bc;

  for (int tile = 0; tile < num_tiles; tile++) {
    __shared__ float sK[Bc * 128];
    __shared__ float sV[Bc * 128];

    // All 128 threads cooperate to load K and V tiles into shared memory
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

    // Process each KV position in this tile
    for (int p = 0; p < Bc; p++) {
      int k_pos = tile * Bc + p;
      if (k_pos >= seqlen) break;

      // Load K values: float4 at lane offset within this row
      float4 k_val = reinterpret_cast<float4*>(&sK[p * d])[lane];

      // Dot product: sum of 4 products within this thread
      float partial = q_val.x * k_val.x + q_val.y * k_val.y
                    + q_val.z * k_val.z + q_val.w * k_val.w;

      // Warp-level reduce: all 32 threads in warp get the total
      float total = warpReduceSum(partial);
      float score = total * sm_scale;

      // Online softmax update (same m/s/score for all threads in warp)
      float new_m = fmaxf(m, score);
      float rescale = expf(m - new_m);
      float p_val = expf(score - new_m);
      s = s * rescale + p_val;
      m = new_m;

      // Load V values: float4 at lane offset
      float4 v_val = reinterpret_cast<float4*>(&sV[p * d])[lane];

      // Update accumulator (4 elements per thread)
      acc_x = acc_x * rescale + p_val * v_val.x;
      acc_y = acc_y * rescale + p_val * v_val.y;
      acc_z = acc_z * rescale + p_val * v_val.z;
      acc_w = acc_w * rescale + p_val * v_val.w;
    }
    __syncthreads();
  }

  // Write output
  if (q_row < seqlen) {
    float inv_s = 1.0f / s;
    float4 result;
    result.x = acc_x * inv_s;
    result.y = acc_y * inv_s;
    result.z = acc_z * inv_s;
    result.w = acc_w * inv_s;
    reinterpret_cast<float4*>(O)[q_row * (d / 4) + lane] = result;
  }
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

  if (dim == 128) {
    // Br=4 kernel: 4 query rows per block, warp-level reduction, float4 per thread
    int Gc = bs * head;
    int Gr = (seqlen + 3) / 4;  // ceil(seqlen / Br)

    dim3 grid(Gc, Gr);
    dim3 block(128);  // 4 warps x 32 threads

    flash_attention_br4_kernel<Bc><<<grid, block>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), seqlen, stride_head, dim, sm_scale);
  } else {
    // Fallback: Br=1 kernel (for dim=64 and other dims)
    int Gc = bs * head;
    int Gr = seqlen;

    dim3 grid(Gc, Gr);
    int block_size = dim < 32 ? 32 : dim;
    dim3 block(block_size);

    flash_attention_br1_kernel<Bc><<<grid, block>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), seqlen, stride_head, dim, sm_scale);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("flash_attention", &flash_attention_cuda);
}