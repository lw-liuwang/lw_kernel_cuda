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

// Histogram: per-warp-group privatization to reduce shared memory atomic contention.
// Groups of 4 warps (128 threads) share a private histogram segment, then merge.
// This reduces atomic contention from 256 threads → 2 groups → 2x fewer collisions.
__global__ void __launch_bounds__(256)
histogram_kernel(const float* x, int* bins, int n,
                                  int64_t num_bins, double min_val, double max_val) {
  // Two warp-group segments (warps 0-3 share segment 0, warps 4-7 share segment 1)
  constexpr int SEGMENTS = 2;
  int warp_id = threadIdx.x / 32;
  int seg_id = warp_id / (32 / SEGMENTS);  // which segment this warp uses
  __shared__ int shared_bins[SEGMENTS][1024];

  // Initialize this thread's segment
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    shared_bins[0][i] = 0;
    if (SEGMENTS > 1) shared_bins[1][i] = 0;
  }
  __syncthreads();

  float range = max_val - min_val;
  float inv_bin = num_bins / range;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float val = x[i];
    if (val >= min_val && val < max_val) {
      int bin = (int)((val - min_val) * inv_bin);
      if (bin >= num_bins) bin = num_bins - 1;
      atomicAdd(&shared_bins[seg_id][bin], 1);
    }
  }
  __syncthreads();

  // Merge segments: each thread merges its bins
  if (SEGMENTS > 1) {
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
      shared_bins[0][i] += shared_bins[1][i];
    }
    __syncthreads();
  }

  // Write to global
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    atomicAdd(&bins[i], shared_bins[0][i]);
  }
}

torch::Tensor histogram_cuda(torch::Tensor x, int64_t num_bins, double min_val, double max_val) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dtype() == at::kFloat);

  int n = x.numel();
  if (n == 0) return torch::zeros({num_bins}, at::dtype(at::kInt).device(at::kCUDA));
  TORCH_CHECK(num_bins <= 1024, "histogram supports at most 1024 bins, got ", num_bins);
  auto bins = torch::zeros({num_bins}, at::dtype(at::kInt).device(at::kCUDA));

  int block_size = 256;
  int grid_size = min((n + block_size - 1) / block_size, 1024);

  histogram_kernel<<<grid_size, block_size>>>(
    x.data_ptr<float>(), bins.data_ptr<int>(), n, num_bins, min_val, max_val);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bins;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("histogram", &histogram_cuda);
}