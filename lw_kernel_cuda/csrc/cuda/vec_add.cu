#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ==================== 输入检查宏 ==================== */
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

/* ==================== GPU Kernel ==================== */
/**
 * VecAdd with float4 vectorized loads/stores.
 * Each thread processes 4 elements per stride, stride = total_threads * 4.
 */
__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // float4 vectorized portion
  const float4* a4 = reinterpret_cast<const float4*>(a);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  float4* c4 = reinterpret_cast<float4*>(c);
  int n4 = n / 4;

  for (int i = idx; i < n4; i += stride) {
    float4 va = a4[i];
    float4 vb = b4[i];
    c4[i] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
  }

  // Tail elements (n % 4 != 0)
  int tail_start = n4 * 4;
  for (int i = tail_start + idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

/* ==================== Host 封装函数 ==================== */
torch::Tensor vec_add_cuda(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(a.sizes() == b.sizes());

  auto out = torch::empty_like(a);
  int n = a.numel();
  if (n == 0) return out;

  // float4 means each thread processes 4 elements, so grid_size is naturally smaller
  // Use larger block for better occupancy, no hard cap on grid_size
  int block_size = 256;
  int grid_size = (n / 4 + block_size - 1) / block_size;  // based on float4 element count
  if (grid_size < 1) grid_size = 1;
  // Cap grid_size to avoid launching too many blocks
  if (grid_size > 1024) grid_size = 1024;

  vec_add_kernel<<<grid_size, block_size>>>(
    a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m) {
  m.impl("vec_add", &vec_add_cuda);
}