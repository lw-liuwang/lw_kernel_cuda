#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C" {
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C",
      NULL,
      -1,
      NULL,
  };
  return PyModule_Create(&module_def);
}
}

// Define all operator schemas in one place.
// Each .cu file provides a TORCH_LIBRARY_IMPL(lw_kernel_cuda, CUDA, m)
// that maps these schemas to actual kernel implementations.
TORCH_LIBRARY(lw_kernel_cuda, m) {
  // Phase 1: VecAdd
  m.def("vec_add(Tensor a, Tensor b) -> Tensor");

  // Phase 2: Reduce
  m.def("reduce(Tensor x) -> Tensor");

  // Phase 3: Softmax
  m.def("softmax(Tensor x) -> Tensor");

  // Phase 4: RMSNorm
  m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");

  // Phase 5: Transpose
  m.def("transpose(Tensor x) -> Tensor");

  // Phase 6: MatMul
  m.def("matmul(Tensor a, Tensor b) -> Tensor");

  // Phase 7: Conv2D
  m.def("conv2d(Tensor x, Tensor weight, Tensor bias, int stride, int padding) -> Tensor");

  // Phase 8: Histogram
  m.def("histogram(Tensor x, int num_bins, float min_val, float max_val) -> Tensor");

  // Phase 9: PrefixSum
  m.def("prefix_sum(Tensor x) -> Tensor");

  // Phase 10: FlashAttention
  m.def("flash_attention(Tensor q, Tensor k, Tensor v) -> Tensor");
}