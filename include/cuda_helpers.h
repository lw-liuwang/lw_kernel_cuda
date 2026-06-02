#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) throw std::runtime_error(cudaGetErrorString(code));
  }
}