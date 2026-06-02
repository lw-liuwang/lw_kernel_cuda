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

/**
 * Inclusive scan using Brent-Kung work-efficient algorithm in shared memory.
 * Each block processes exactly blockDim.x elements.
 *
 * After this kernel, shared[tid] contains the inclusive prefix sum for each element.
 * The block's total sum is stored to block_sums[blockIdx.x] (for multi-block chaining).
 *
 * Brent-Kung uses O(n) work vs Kogge-Stone's O(n log n), reducing redundant additions.
 */
__global__ void scan_blocks_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    float* __restrict__ block_sums,
                                    int n) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int global_idx = blockIdx.x * blockDim.x + tid;

  // Load data into shared memory
  shared[tid] = (global_idx < n) ? in[global_idx] : 0.0f;
  __syncthreads();

  // ----- Brent-Kung up-sweep (reduction tree) -----
  // Each step doubles the stride; only elements at stride-1, 2*stride-1, etc. participate
  #pragma unroll
  for (int d = 0; d < 8; d++) {  // log2(256) = 8
    int step = 2 << d;           // 2, 4, 8, 16, 32, 64, 128, 256
    int half = step >> 1;        // 1, 2, 4, 8, 16, 32, 64, 128
    if ((tid & (step - 1)) == (step - 1)) {
      shared[tid] += shared[tid - half];
    }
    __syncthreads();
  }

  // Store block sum (last element after up-sweep = total sum of block's chunk)
  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = shared[blockDim.x - 1];
  }

  // ----- Brent-Kung down-sweep (build exclusive scan from partial sums) -----
  if (tid == 0) {
    shared[blockDim.x - 1] = 0.0f;  // reset last element to 0 (exclusive base)
  }
  __syncthreads();

  #pragma unroll
  for (int d = 7; d >= 0; d--) {
    int step = 2 << d;
    int half = step >> 1;
    if ((tid & (step - 1)) == (step - 1)) {
      float t = shared[tid - half];
      shared[tid - half] = shared[tid];
      shared[tid] += t;
    }
    __syncthreads();
  }

  // Convert exclusive scan to inclusive: result[tid] = exclusive[tid] + original[tid]
  // We saved original[tid]... but shared[tid] was overwritten by the scan!
  // Need to re-read original from global memory (aliased by n checks).
  // Actually, we need the original value. Approach: load into register before scan.
  // But we already overwrote shared... Instead, just do inclusive scan differently.
  //
  // Actually, the down-sweep produces exclusive scan = sum of elements BEFORE tid.
  // To get inclusive: shared[tid] = exclusive + original = shared[tid] + in[global_idx]
  // But only if global_idx < n.
  //
  // Simple correction: we stored original in shared[0..blockDim.x-1] before scan.
  // After Brent-Kung down-sweep, shared[tid] = exclusive scan.
  // To convert to inclusive scan: shared[tid] += original_value.
  // We can reload original from global mem: in[global_idx] (still valid, global read-only).
  if (global_idx < n) {
    // shared[tid] is currently exclusive scan (sum of elements BEFORE this element)
    // Add the element itself to get inclusive scan
    shared[tid] += in[global_idx];
  }
  __syncthreads();

  // Write output
  if (global_idx < n) {
    out[global_idx] = shared[tid];
  }
}

/**
 * Single-block exclusive scan of block_sums array.
 * After this, block_sums[i] = sum of original block_sums[0..i-1] (exclusive prefix).
 * Used to compute per-block offsets for multi-block chaining.
 */
__global__ void scan_block_sums_kernel(float* block_sums, int grid_size) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;

  shared[tid] = (tid < grid_size) ? block_sums[tid] : 0.0f;
  __syncthreads();

  // Brent-Kung up-sweep
  #pragma unroll
  for (int d = 0; d < 8; d++) {
    int step = 2 << d;
    int half = step >> 1;
    if ((tid & (step - 1)) == (step - 1)) {
      shared[tid] += shared[tid - half];
    }
    __syncthreads();
  }

  // Brent-Kung down-sweep (produces exclusive scan)
  if (tid == 0) {
    shared[blockDim.x - 1] = 0.0f;
  }
  __syncthreads();

  #pragma unroll
  for (int d = 7; d >= 0; d--) {
    int step = 2 << d;
    int half = step >> 1;
    if ((tid & (step - 1)) == (step - 1)) {
      float t = shared[tid - half];
      shared[tid - half] = shared[tid];
      shared[tid] += t;
    }
    __syncthreads();
  }

  // Write back exclusive prefix sums
  if (tid < grid_size) {
    block_sums[tid] = shared[tid];
  }
}

/**
 * Add per-block prefix offsets to each element of the output.
 * After this, out[i] = inclusive_scan_within_block(i) + sum_of_previous_blocks.
 */
__global__ void add_block_sums_kernel(float* __restrict__ out,
                                       const float* __restrict__ block_sums,
                                       int n) {
  int tid = threadIdx.x;
  int global_idx = blockIdx.x * blockDim.x + tid;
  if (global_idx >= n) return;

  out[global_idx] += block_sums[blockIdx.x];
}

torch::Tensor prefix_sum_cuda(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dtype() == at::kFloat);

  int n = x.numel();
  auto out = torch::empty_like(x);
  if (n == 0) return out;

  // block_size must be power of 2 for Brent-Kung scan
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  // Allocate block_sums array for multi-block chaining
  auto block_sums = torch::empty({grid_size}, x.options());

  // Step 1: Each block does inclusive scan on its chunk
  // Also writes block sum to block_sums
  scan_blocks_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
    x.data_ptr<float>(), out.data_ptr<float>(),
    block_sums.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (grid_size > 1) {
    // Step 2: Single block scans the block_sums array (exclusive)
    scan_block_sums_kernel<<<1, block_size, block_size * sizeof(float)>>>(
      block_sums.data_ptr<float>(), grid_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 3: Add cumulative block offsets to each element
    add_block_sums_kernel<<<grid_size, block_size>>>(
      out.data_ptr<float>(), block_sums.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("prefix_sum", &prefix_sum_cuda);
}