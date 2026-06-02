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

  // Convert exclusive scan to inclusive: add the original value
  if (global_idx < n) {
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
 * Uses grid-stride loop to handle arbitrary grid_size (not limited to blockDim.x).
 *
 * After this, block_sums[i] = sum of original block_sums[0..i-1] (exclusive prefix).
 * Used to compute per-block offsets for multi-block chaining.
 *
 * For large grid_size (> blockDim.x), processes the array in chunks of blockDim.x
 * and carries the cumulative sum across chunks.
 */
__global__ void scan_block_sums_kernel(float* block_sums, int grid_size) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int blockDim_x = blockDim.x;

  float carry = 0.0f;

  for (int chunk = 0; chunk * blockDim_x < grid_size; chunk++) {
    int base = chunk * blockDim_x;

    // Load this chunk's elements
    shared[tid] = (base + tid < grid_size) ? block_sums[base + tid] : 0.0f;
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

    // Save chunk total before down-sweep overwrites it
    float chunk_total = shared[blockDim_x - 1];

    // Brent-Kung down-sweep (produces exclusive scan within this chunk)
    if (tid == 0) {
      shared[blockDim_x - 1] = 0.0f;
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

    // Write back: local exclusive scan + carry from previous chunks
    if (base + tid < grid_size) {
      block_sums[base + tid] = shared[tid] + carry;
    }
    __syncthreads();

    // Update carry for next chunk
    carry += chunk_total;
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
  scan_blocks_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
    x.data_ptr<float>(), out.data_ptr<float>(),
    block_sums.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (grid_size > 1) {
    // Step 2: Scan the block_sums array (exclusive) — supports arbitrary grid_size
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