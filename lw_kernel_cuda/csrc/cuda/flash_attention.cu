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

// ----------------------------------------------------------------
// Br=4 warp-reduce kernel (for d=64): 4 query rows per block
// ----------------------------------------------------------------
/**
 * Architecture: same as d=128 Br=4 but with float2 per thread (d=64, 64/32=2)
 *   - blockDim = 128 (4 warps x 32 threads)
 *   - Each warp handles 1 query row → Br = 4
 *   - grid = (B*H, seqlen/Br)
 *   - Bc = 32: K/V tile of 32 positions
 *   - Each thread handles d/warpSize = 2 elements via float2
 *   - Dot product: per-thread float2 dot → warpReduceSum (pure shuffle, no smem)
 *   - Online softmax per-warp state
 */
template <int Bc>
__global__ void flash_attention_br4_kernel_d64(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen, int stride_head, int d, float sm_scale) {

  int tid = threadIdx.x;
  int warp = tid >> 5;       // warp = 0..3, each handles one query row
  int lane = tid & 31;       // lane = 0..31, each handles 2 d-elements (d=64)

  int head_base = blockIdx.x * stride_head;
  int q_base_row = blockIdx.y * 4;

  Q += head_base; K += head_base; V += head_base; O += head_base;

  int q_row = q_base_row + warp;

  // Each thread handles 2 elements in the d dimension (d=64, 64/32=2)

  // Load Q values (float2)
  float2 q_val;
  if (q_row < seqlen) {
    q_val = reinterpret_cast<const float2*>(Q)[q_row * (d / 2) + lane];
  } else {
    q_val = make_float2(0.0f, 0.0f);
  }

  // Online softmax state (per-thread, shared across warp via shuffle broadcast)
  float m = -INFINITY;
  float s = 0.0f;
  float acc_x = 0.0f, acc_y = 0.0f;

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

      // Load K values: float2 at lane offset within this row
      float2 k_val = reinterpret_cast<float2*>(&sK[p * d])[lane];

      // Dot product: sum of 2 products within this thread
      float partial = q_val.x * k_val.x + q_val.y * k_val.y;

      // Warp-level reduce: all 32 threads in warp get the total
      float total = warpReduceSum(partial);
      float score = total * sm_scale;

      // Online softmax update (same m/s/score for all threads in warp)
      float new_m = fmaxf(m, score);
      float rescale = expf(m - new_m);
      float p_val = expf(score - new_m);
      s = s * rescale + p_val;
      m = new_m;

      // Load V values: float2 at lane offset
      float2 v_val = reinterpret_cast<float2*>(&sV[p * d])[lane];

      // Update accumulator (2 elements per thread)
      acc_x = acc_x * rescale + p_val * v_val.x;
      acc_y = acc_y * rescale + p_val * v_val.y;
    }
    __syncthreads();
  }

  // Write output
  if (q_row < seqlen) {
    float inv_s = 1.0f / s;
    float2 result;
    result.x = acc_x * inv_s;
    result.y = acc_y * inv_s;
    reinterpret_cast<float2*>(O)[q_row * (d / 2) + lane] = result;
  }
}

// ----------------------------------------------------------------
// Br=8 warp-reduce kernel (for d=128): 8 query rows per block
// ----------------------------------------------------------------
/**
 * Architecture:
 *   - blockDim = 128 (4 warps x 32 threads)
 *   - Each warp handles 2 query rows → Br = 8
 *   - grid = (B*H, seqlen/8)
 *   - Bc = 32: K/V tile of 32 positions
 *   - Each thread handles d/warpSize = 4 elements via float4
 *   - Dot product: per-thread float4 dot → warpReduceSum (pure shuffle, no smem)
 *   - Online softmax per-warp state (2 rows per warp)
 *
 * Compared to Br=4: halves grid size, each K/V global read services 8 rows.
 */
template <int Bc>
__global__ void flash_attention_br8_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen, int stride_head, int d, float sm_scale) {

  int tid = threadIdx.x;
  int warp = tid >> 5;       // warp = 0..3, each handles two query rows
  int lane = tid & 31;       // lane = 0..31, each handles 4 d-elements (d=128)

  int head_base = blockIdx.x * stride_head;
  int q_base_row = blockIdx.y * 8;  // Br = 8

  Q += head_base; K += head_base; V += head_base; O += head_base;

  int q_row0 = q_base_row + warp * 2;
  int q_row1 = q_base_row + warp * 2 + 1;

  // Load Q values for both rows (float4)
  float4 q0_val, q1_val;
  if (q_row0 < seqlen) {
    q0_val = reinterpret_cast<const float4*>(Q)[q_row0 * (d / 4) + lane];
  } else {
    q0_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  if (q_row1 < seqlen) {
    q1_val = reinterpret_cast<const float4*>(Q)[q_row1 * (d / 4) + lane];
  } else {
    q1_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  // Online softmax state for both rows
  float m0 = -INFINITY, m1 = -INFINITY;
  float s0 = 0.0f, s1 = 0.0f;
  float acc0_x = 0.0f, acc0_y = 0.0f, acc0_z = 0.0f, acc0_w = 0.0f;
  float acc1_x = 0.0f, acc1_y = 0.0f, acc1_z = 0.0f, acc1_w = 0.0f;

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

      // Dot products for both rows
      float partial0 = q0_val.x * k_val.x + q0_val.y * k_val.y
                     + q0_val.z * k_val.z + q0_val.w * k_val.w;
      float partial1 = q1_val.x * k_val.x + q1_val.y * k_val.y
                     + q1_val.z * k_val.z + q1_val.w * k_val.w;

      // Warp-level reduce: both shuffles, no smem/sync
      float total0 = warpReduceSum(partial0);
      float total1 = warpReduceSum(partial1);

      float score0 = total0 * sm_scale;
      float score1 = total1 * sm_scale;

      // Online softmax update for row 0
      float new_m0 = fmaxf(m0, score0);
      float rescale0 = expf(m0 - new_m0);
      float p_val0 = expf(score0 - new_m0);
      s0 = s0 * rescale0 + p_val0;
      m0 = new_m0;

      // Online softmax update for row 1
      float new_m1 = fmaxf(m1, score1);
      float rescale1 = expf(m1 - new_m1);
      float p_val1 = expf(score1 - new_m1);
      s1 = s1 * rescale1 + p_val1;
      m1 = new_m1;

      // Load V values: float4 at lane offset
      float4 v_val = reinterpret_cast<float4*>(&sV[p * d])[lane];

      // Update accumulators for both rows (same V, different p_val/rescale)
      acc0_x = acc0_x * rescale0 + p_val0 * v_val.x;
      acc0_y = acc0_y * rescale0 + p_val0 * v_val.y;
      acc0_z = acc0_z * rescale0 + p_val0 * v_val.z;
      acc0_w = acc0_w * rescale0 + p_val0 * v_val.w;

      acc1_x = acc1_x * rescale1 + p_val1 * v_val.x;
      acc1_y = acc1_y * rescale1 + p_val1 * v_val.y;
      acc1_z = acc1_z * rescale1 + p_val1 * v_val.z;
      acc1_w = acc1_w * rescale1 + p_val1 * v_val.w;
    }
    __syncthreads();
  }

  // Write output for both rows
  if (q_row0 < seqlen) {
    float inv_s = 1.0f / s0;
    float4 result;
    result.x = acc0_x * inv_s;
    result.y = acc0_y * inv_s;
    result.z = acc0_z * inv_s;
    result.w = acc0_w * inv_s;
    reinterpret_cast<float4*>(O)[q_row0 * (d / 4) + lane] = result;
  }
  if (q_row1 < seqlen) {
    float inv_s = 1.0f / s1;
    float4 result;
    result.x = acc1_x * inv_s;
    result.y = acc1_y * inv_s;
    result.z = acc1_z * inv_s;
    result.w = acc1_w * inv_s;
    reinterpret_cast<float4*>(O)[q_row1 * (d / 4) + lane] = result;
  }
}

// ----------------------------------------------------------------
// Br=8 warp-reduce kernel (for d=64): 8 query rows per block
// ----------------------------------------------------------------
template <int Bc>
__global__ void flash_attention_br8_kernel_d64(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen, int stride_head, int d, float sm_scale) {

  int tid = threadIdx.x;
  int warp = tid >> 5;       // warp = 0..3, each handles two query rows
  int lane = tid & 31;       // lane = 0..31, each handles 2 d-elements (d=64)

  int head_base = blockIdx.x * stride_head;
  int q_base_row = blockIdx.y * 8;  // Br = 8

  Q += head_base; K += head_base; V += head_base; O += head_base;

  int q_row0 = q_base_row + warp * 2;
  int q_row1 = q_base_row + warp * 2 + 1;

  // Load Q values for both rows (float2)
  float2 q0_val, q1_val;
  if (q_row0 < seqlen) {
    q0_val = reinterpret_cast<const float2*>(Q)[q_row0 * (d / 2) + lane];
  } else {
    q0_val = make_float2(0.0f, 0.0f);
  }
  if (q_row1 < seqlen) {
    q1_val = reinterpret_cast<const float2*>(Q)[q_row1 * (d / 2) + lane];
  } else {
    q1_val = make_float2(0.0f, 0.0f);
  }

  // Online softmax state for both rows
  float m0 = -INFINITY, m1 = -INFINITY;
  float s0 = 0.0f, s1 = 0.0f;
  float acc0_x = 0.0f, acc0_y = 0.0f;
  float acc1_x = 0.0f, acc1_y = 0.0f;

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

      float2 k_val = reinterpret_cast<float2*>(&sK[p * d])[lane];

      float partial0 = q0_val.x * k_val.x + q0_val.y * k_val.y;
      float partial1 = q1_val.x * k_val.x + q1_val.y * k_val.y;

      float total0 = warpReduceSum(partial0);
      float total1 = warpReduceSum(partial1);

      float score0 = total0 * sm_scale;
      float score1 = total1 * sm_scale;

      // Online softmax row 0
      float new_m0 = fmaxf(m0, score0);
      float rescale0 = expf(m0 - new_m0);
      float p_val0 = expf(score0 - new_m0);
      s0 = s0 * rescale0 + p_val0;
      m0 = new_m0;

      // Online softmax row 1
      float new_m1 = fmaxf(m1, score1);
      float rescale1 = expf(m1 - new_m1);
      float p_val1 = expf(score1 - new_m1);
      s1 = s1 * rescale1 + p_val1;
      m1 = new_m1;

      // Load V values
      float2 v_val = reinterpret_cast<float2*>(&sV[p * d])[lane];

      // Update accumulators for both rows
      acc0_x = acc0_x * rescale0 + p_val0 * v_val.x;
      acc0_y = acc0_y * rescale0 + p_val0 * v_val.y;

      acc1_x = acc1_x * rescale1 + p_val1 * v_val.x;
      acc1_y = acc1_y * rescale1 + p_val1 * v_val.y;
    }
    __syncthreads();
  }

  // Write output for both rows
  if (q_row0 < seqlen) {
    float inv_s = 1.0f / s0;
    float2 result;
    result.x = acc0_x * inv_s;
    result.y = acc0_y * inv_s;
    reinterpret_cast<float2*>(O)[q_row0 * (d / 2) + lane] = result;
  }
  if (q_row1 < seqlen) {
    float inv_s = 1.0f / s1;
    float2 result;
    result.x = acc1_x * inv_s;
    result.y = acc1_y * inv_s;
    reinterpret_cast<float2*>(O)[q_row1 * (d / 2) + lane] = result;
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
    // Br=8 kernel: 8 query rows per block, warp-level reduction, float4 per thread
    int Gc = bs * head;
    int Gr = (seqlen + 7) / 8;  // ceil(seqlen / Br=8)

    dim3 grid(Gc, Gr);
    dim3 block(128);  // 4 warps x 32 threads

    flash_attention_br8_kernel<Bc><<<grid, block>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), seqlen, stride_head, dim, sm_scale);
  } else if (dim == 64) {
    // Br=8 kernel for d=64: float2 per thread, warp-level reduction
    int Gc = bs * head;
    int Gr = (seqlen + 7) / 8;  // ceil(seqlen / Br=8)

    dim3 grid(Gc, Gr);
    dim3 block(128);  // 4 warps x 32 threads

    flash_attention_br8_kernel_d64<Bc><<<grid, block>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), seqlen, stride_head, dim, sm_scale);
  } else {
    // Fallback: Br=1 kernel (for other dims) — use Bc=32 to stay under 48KB shared mem limit
    int Gc = bs * head;
    int Gr = seqlen;

    dim3 grid(Gc, Gr);
    int block_size = dim < 32 ? 32 : dim;
    dim3 block(block_size);

    flash_attention_br1_kernel<32><<<grid, block>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      out.data_ptr<float>(), seqlen, stride_head, dim, sm_scale);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("flash_attention", &flash_attention_cuda);
}