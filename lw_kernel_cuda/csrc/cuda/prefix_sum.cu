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
 * Inclusive scan using up-sweep/down-sweep work-efficient algorithm.
 * Each block processes 2 * blockDim.x elements (2 elements per thread).
 *
 * After this kernel, shared[2*tid] and shared[2*tid+1] contain the inclusive
 * prefix sums. The block's total sum is stored to block_sums[blockIdx.x].
 *
 * Using 2 elements/thread halves the grid size vs 1 element/thread version,
 * reducing kernel launch overhead and improving throughput for large inputs.
 */
__global__ void scan_blocks_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    float* __restrict__ block_sums,
                                    int n) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int blockDim_x = blockDim.x;
  int m = 2 * blockDim_x;  // elements per block
  int base = blockIdx.x * m;

  // Load 2 elements per thread into shared memory
  shared[2 * tid] = (base + 2 * tid < n) ? in[base + 2 * tid] : 0.0f;
  shared[2 * tid + 1] = (base + 2 * tid + 1 < n) ? in[base + 2 * tid + 1] : 0.0f;
  __syncthreads();

  // ----- Up-sweep (reduction tree) -----
  for (int stride = 1; stride < m; stride <<= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < m) {
      shared[index] += shared[index - stride];
    }
    __syncthreads();
  }

  // Store block sum (last element after up-sweep = total sum of this chunk)
  if (tid == 0) {
    block_sums[blockIdx.x] = shared[m - 1];
  }

  // ----- Down-sweep (build inclusive scan from partial sums) -----
  // Note: we do NOT zero out shared[m-1] first.
  // This produces inclusive prefix sums directly (per Blelloch "post scan").
  for (int stride = m >> 2; stride > 0; stride >>= 1) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index + stride < m) {
      shared[index + stride] += shared[index];
    }
    __syncthreads();
  }

  // Write output (values are already inclusive scan)
  if (base + 2 * tid < n) {
    out[base + 2 * tid] = shared[2 * tid];
  }
  if (base + 2 * tid + 1 < n) {
    out[base + 2 * tid + 1] = shared[2 * tid + 1];
  }
}

/**
 * Exclusive scan of block_sums array using proper Blelloch down-sweep.
 * Uses grid-stride loop to handle arbitrary grid_size (not limited to blockDim.x).
 * Each chunk processes 2 * blockDim.x elements for better efficiency.
 *
 * After this kernel, block_sums[i] = sum of original block_sums[0..i-1] (exclusive).
 */
__global__ void scan_block_sums_kernel(float* block_sums, int grid_size) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int blockDim_x = blockDim.x;
  int chunk_size = 2 * blockDim_x;

  float carry = 0.0f;

  for (int chunk = 0; chunk * chunk_size < grid_size; chunk++) {
    int base = chunk * chunk_size;

    // Load 2 elements per thread
    shared[2 * tid] = (base + 2 * tid < grid_size) ? block_sums[base + 2 * tid] : 0.0f;
    shared[2 * tid + 1] = (base + 2 * tid + 1 < grid_size) ? block_sums[base + 2 * tid + 1] : 0.0f;
    __syncthreads();

    // ----- Up-sweep (reduction tree) -----
    for (int stride = 1; stride < chunk_size; stride <<= 1) {
      int index = (tid + 1) * stride * 2 - 1;
      if (index < chunk_size) {
        shared[index] += shared[index - stride];
      }
      __syncthreads();
    }

    // Save total sum of this chunk, then set last element to 0
    // (key difference from inclusive post-scan: this enables exclusive scan output)
    float chunk_total = shared[chunk_size - 1];
    if (tid == 0) {
      shared[chunk_size - 1] = 0.0f;
    }
    __syncthreads();

    // ----- Exclusive down-sweep (Blelloch) -----
    // Uses copy-and-add: t = shared[left]; shared[left] = shared[right]; shared[right] += t;
    for (int stride = chunk_size >> 1; stride > 0; stride >>= 1) {
      int index = (tid + 1) * stride * 2 - 1;
      int right = index;
      int left = right - stride;
      if (right < chunk_size) {
        float t = shared[left];
        shared[left] = shared[right];
        shared[right] += t;
      }
      __syncthreads();
    }

    // Write back: exclusive scan values within this chunk + carry from previous chunks
    if (base + 2 * tid < grid_size) {
      block_sums[base + 2 * tid] = shared[2 * tid] + carry;
    }
    if (base + 2 * tid + 1 < grid_size) {
      block_sums[base + 2 * tid + 1] = shared[2 * tid + 1] + carry;
    }
    __syncthreads();

    // Update carry for next chunk
    carry += chunk_total;
  }
}

/**
 * Add per-block prefix offsets to each element of the output.
 * Each block handles 2 * blockDim.x elements (matching scan_blocks_kernel).
 */
__global__ void add_block_sums_kernel(float* __restrict__ out,
                                       const float* __restrict__ block_sums,
                                       int n) {
  int tid = threadIdx.x;
  int blockDim_x = blockDim.x;
  int base = blockIdx.x * (2 * blockDim_x);

  if (base + 2 * tid < n) {
    out[base + 2 * tid] += block_sums[blockIdx.x];
  }
  if (base + 2 * tid + 1 < n) {
    out[base + 2 * tid + 1] += block_sums[blockIdx.x];
  }
}

torch::Tensor prefix_sum_cuda(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dtype() == at::kFloat);

  int n = x.numel();
  auto out = torch::empty_like(x);
  if (n == 0) return out;

  // Each block processes 2 * block_size elements (2 per thread)
  int block_size = 256;
  int elements_per_block = 2 * block_size;
  int grid_size = (n + elements_per_block - 1) / elements_per_block;

  // Shared memory: chunk_size = 2 * block_size elements
  int shared_size = elements_per_block * sizeof(float);

  // Allocate block_sums array for multi-block chaining
  auto block_sums = torch::empty({grid_size}, x.options());

  // Step 1: Each block does inclusive scan on its chunk (2 elements/thread)
  scan_blocks_kernel<<<grid_size, block_size, shared_size>>>(
    x.data_ptr<float>(), out.data_ptr<float>(),
    block_sums.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (grid_size > 1) {
    // Step 2: Scan the block_sums array (exclusive) — supports arbitrary grid_size
    scan_block_sums_kernel<<<1, block_size, shared_size>>>(
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