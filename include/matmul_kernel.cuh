#ifndef MATMUL_KERNEL_CUH
#define MATMUL_KERNEL_CUH

#include <cuda_runtime.h>

constexpr int MM_WARP_SIZE = 32;

/**
 * MatMul kernel with warp tiling, float4 vectorized loads, and bank-conflict-free As layout.
 *
 * Parameters:
 *   BM, BN = block tile size (128x128)
 *   BK = inner reduction dimension tile (16)
 *   WM, WN = warp tile size (64x64)
 *   WNITER = N sub-iterations per warp (4)
 *   TM, TN = per-thread register tile (8x4)
 *   NUM_THREADS = threads per block (128 = 4 warps)
 */
template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_kernel(int M, int N, int K,
              const float* __restrict__ A,
              const float* __restrict__ B,
              float* __restrict__ C) {
  const int c_row = blockIdx.y;
  const int c_col = blockIdx.x;

  const int warp_idx = threadIdx.x / MM_WARP_SIZE;
  const int warp_col = warp_idx % (BN / WN);
  const int warp_row = warp_idx / (BN / WN);

  constexpr int WMITER = (WM * WN) / (MM_WARP_SIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  const int thread_id_in_warp = threadIdx.x % MM_WARP_SIZE;
  const int thread_col_in_warp = thread_id_in_warp % (WSUBN / TN);
  const int thread_row_in_warp = thread_id_in_warp / (WSUBN / TN);

  __shared__ float As[BM * (BK + 1)];  // +1 padding for bank conflict avoidance
  __shared__ float Bs[BK * BN];

  float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
  float reg_m[WMITER * TM];
  float reg_n[WNITER * TN];

  const int block_row_start = c_row * BM;
  const int block_col_start = c_col * BN;
  A += block_row_start * K;
  B += block_col_start;
  C += (block_row_start + warp_row * WM) * N + block_col_start + warp_col * WN;

  const int inner_row_a = threadIdx.x / (BK / 4);
  const int inner_col_a = threadIdx.x % (BK / 4);
  constexpr int row_stride_a = (NUM_THREADS * 4) / BK;

  const int inner_row_b = threadIdx.x / (BN / 4);
  const int inner_col_b = threadIdx.x % (BN / 4);
  constexpr int row_stride_b = NUM_THREADS / (BN / 4);

  for (int bk = 0; bk < K; bk += BK) {
    // Load As tile
    #pragma unroll
    for (int off = 0; off < BM; off += row_stride_a) {
      int row = inner_row_a + off;
      if (block_row_start + row < M) {
        const float* a_row = &A[row * K];
        bool a_aligned = (((uintptr_t)(a_row + inner_col_a * 4)) & 15) == 0;
        if (a_aligned && bk + inner_col_a * 4 + 3 < K) {
          float4 tmp = reinterpret_cast<const float4*>(a_row + inner_col_a * 4)[0];
          As[row * (BK + 1) + inner_col_a * 4 + 0] = tmp.x;
          As[row * (BK + 1) + inner_col_a * 4 + 1] = tmp.y;
          As[row * (BK + 1) + inner_col_a * 4 + 2] = tmp.z;
          As[row * (BK + 1) + inner_col_a * 4 + 3] = tmp.w;
        } else {
          for (int kk = inner_col_a * 4; kk < inner_col_a * 4 + 4 && kk < BK; kk++) {
            As[row * (BK + 1) + kk] = (bk + kk < K) ? a_row[kk] : 0.0f;
          }
        }
      } else {
        for (int kk = inner_col_a * 4; kk < inner_col_a * 4 + 4 && kk < BK; kk++) {
          As[row * (BK + 1) + kk] = 0.0f;
        }
      }
    }

    // Load Bs tile
    #pragma unroll
    for (int off = 0; off < BK; off += row_stride_b) {
      int row = inner_row_b + off;
      if (bk + row < K) {
        const float* b_row = &B[row * N];
        bool b_aligned = (((uintptr_t)(b_row + inner_col_b * 4)) & 15) == 0;
        if (b_aligned && block_col_start + inner_col_b * 4 + 3 < N) {
          reinterpret_cast<float4*>(&Bs[row * BN + inner_col_b * 4])[0] =
              reinterpret_cast<const float4*>(b_row + inner_col_b * 4)[0];
        } else {
          for (int nn = inner_col_b * 4; nn < inner_col_b * 4 + 4 && nn < BN; nn++) {
            Bs[row * BN + nn] = (block_col_start + nn < N) ? b_row[nn] : 0.0f;
          }
        }
      } else {
        for (int nn = inner_col_b * 4; nn < inner_col_b * 4 + 4 && nn < BN; nn++) {
          Bs[row * BN + nn] = 0.0f;
        }
      }
    }
    __syncthreads();

    // Compute BK iterations
    #pragma unroll
    for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
      #pragma unroll
      for (int wsri = 0; wsri < WMITER; wsri++) {
        #pragma unroll
        for (int i = 0; i < TM; i++) {
          reg_m[wsri * TM + i] =
              As[(warp_row * WM + wsri * WSUBM +
                  thread_row_in_warp * TM + i) * (BK + 1) + dot_idx];
        }
      }
      #pragma unroll
      for (int wsci = 0; wsci < WNITER; wsci++) {
        #pragma unroll
        for (int i = 0; i < TN; i++) {
          reg_n[wsci * TN + i] =
              Bs[dot_idx * BN + warp_col * WN +
                 wsci * WSUBN + thread_col_in_warp * TN + i];
        }
      }
      #pragma unroll
      for (int wsri = 0; wsri < WMITER; wsri++) {
        #pragma unroll
        for (int wsci = 0; wsci < WNITER; wsci++) {
          #pragma unroll
          for (int rm = 0; rm < TM; rm++) {
            float a_val = reg_m[wsri * TM + rm];
            #pragma unroll
            for (int rn = 0; rn < TN; rn++) {
              thread_results[(wsri * TM + rm) * (WNITER * TN) +
                             wsci * TN + rn] +=
                  a_val * reg_n[wsci * TN + rn];
            }
          }
        }
      }
    }
    __syncthreads();

    A += BK;
    B += BK * N;
  }

  // Write results
  #pragma unroll
  for (int wsri = 0; wsri < WMITER; wsri++) {
    #pragma unroll
    for (int wsci = 0; wsci < WNITER; wsci++) {
      float* C_sub = C + (wsri * WSUBM) * N + wsci * WSUBN;
      int gbase_m = block_row_start + warp_row * WM + wsri * WSUBM + thread_row_in_warp * TM;
      int gbase_n = block_col_start + warp_col * WN + wsci * WSUBN + thread_col_in_warp * TN;

      for (int rm = 0; rm < TM; rm++) {
        int gm = gbase_m + rm;
        if (gm >= M) continue;
        int gn = gbase_n;
        int idx = (wsri * TM + rm) * (WNITER * TN) + wsci * TN;
        float* c_addr = &C_sub[(thread_row_in_warp * TM + rm) * N +
                                thread_col_in_warp * TN];
        bool c_aligned = (((uintptr_t)c_addr) & 15) == 0;
        if (c_aligned && gn + TN <= N) {
          float4 val;
          val.x = thread_results[idx + 0];
          val.y = thread_results[idx + 1];
          val.z = thread_results[idx + 2];
          val.w = thread_results[idx + 3];
          reinterpret_cast<float4*>(c_addr)[0] = val;
        } else {
          for (int rn = 0; rn < TN; rn++) {
            if (gn + rn < N) {
              c_addr[rn] = thread_results[idx + rn];
            }
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Double-buffer matmul kernel with cp.async (requires sm_86+, CUDA 11+)
// -----------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
#include <cuda_pipeline.h>
#endif

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
matmul_kernel_db(int M, int N, int K,
                 const float* __restrict__ A,
                 const float* __restrict__ B,
                 float* __restrict__ C) {
  const int c_row = blockIdx.y;
  const int c_col = blockIdx.x;

  const int warp_idx = threadIdx.x / MM_WARP_SIZE;
  const int warp_col = warp_idx % (BN / WN);
  const int warp_row = warp_idx / (BN / WN);

  constexpr int WMITER = (WM * WN) / (MM_WARP_SIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  const int thread_id_in_warp = threadIdx.x % MM_WARP_SIZE;
  const int thread_col_in_warp = thread_id_in_warp % (WSUBN / TN);
  const int thread_row_in_warp = thread_id_in_warp / (WSUBN / TN);

  // Ping-pong shared memory buffers
  __shared__ float As_ping[BM * (BK + 1)];
  __shared__ float Bs_ping[BK * BN];
  __shared__ float As_pong[BM * (BK + 1)];
  __shared__ float Bs_pong[BK * BN];

  float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
  float reg_m[WMITER * TM];
  float reg_n[WNITER * TN];

  const int block_row_start = c_row * BM;
  const int block_col_start = c_col * BN;

  // Save base pointers for offset-based loads (not modified during loop)
  const float* A_base = A + block_row_start * K;
  const float* B_base = B + block_col_start;
  C += (block_row_start + warp_row * WM) * N + block_col_start + warp_col * WN;

  const int inner_row_a = threadIdx.x / (BK / 4);
  const int inner_col_a = threadIdx.x % (BK / 4);
  constexpr int row_stride_a = (NUM_THREADS * 4) / BK;

  const int inner_row_b = threadIdx.x / (BN / 4);
  const int inner_col_b = threadIdx.x % (BN / 4);
  constexpr int row_stride_b = NUM_THREADS / (BN / 4);

  // ---- Load first tile (bk=0) into ping buffer synchronously ----
  #pragma unroll
  for (int off = 0; off < BM; off += row_stride_a) {
    int row = inner_row_a + off;
    if (block_row_start + row < M) {
      const float* a_row = &A_base[row * K];
      bool a_aligned = (((uintptr_t)(a_row + inner_col_a * 4)) & 15) == 0;
      if (a_aligned && inner_col_a * 4 + 3 < BK) {
        float4 tmp = reinterpret_cast<const float4*>(a_row + inner_col_a * 4)[0];
        As_ping[row * (BK + 1) + inner_col_a * 4 + 0] = tmp.x;
        As_ping[row * (BK + 1) + inner_col_a * 4 + 1] = tmp.y;
        As_ping[row * (BK + 1) + inner_col_a * 4 + 2] = tmp.z;
        As_ping[row * (BK + 1) + inner_col_a * 4 + 3] = tmp.w;
      } else {
        for (int kk = inner_col_a * 4; kk < inner_col_a * 4 + 4 && kk < BK; kk++) {
          As_ping[row * (BK + 1) + kk] = (kk < BK) ? a_row[kk] : 0.0f;
        }
      }
    } else {
      for (int kk = inner_col_a * 4; kk < inner_col_a * 4 + 4 && kk < BK; kk++) {
        As_ping[row * (BK + 1) + kk] = 0.0f;
      }
    }
  }
  #pragma unroll
  for (int off = 0; off < BK; off += row_stride_b) {
    int row = inner_row_b + off;
    if (row < K) {
      const float* b_row = &B_base[row * N];
      bool b_aligned = (((uintptr_t)(b_row + inner_col_b * 4)) & 15) == 0;
      if (b_aligned && block_col_start + inner_col_b * 4 + 3 < N) {
        reinterpret_cast<float4*>(&Bs_ping[row * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4*>(b_row + inner_col_b * 4)[0];
      } else {
        for (int nn = inner_col_b * 4; nn < inner_col_b * 4 + 4 && nn < BN; nn++) {
          Bs_ping[row * BN + nn] = (block_col_start + nn < N) ? b_row[nn] : 0.0f;
        }
      }
    } else {
      for (int nn = inner_col_b * 4; nn < inner_col_b * 4 + 4 && nn < BN; nn++) {
        Bs_ping[row * BN + nn] = 0.0f;
      }
    }
  }
  __syncthreads();

  // Buffer state
  float* As_cur = As_ping;
  float* Bs_cur = Bs_ping;
  float* As_nxt = As_pong;
  float* Bs_nxt = Bs_pong;

  // ---- Main loop with double buffering ----
  for (int bk = 0; bk < K; bk += BK) {
    bool has_next = (bk + BK < K);

    if (has_next) {
      int next_bk = bk + BK;

      // Async load next A tile into As_nxt (4-byte cp.async to avoid alignment issues with stride BK+1)
      #pragma unroll
      for (int off = 0; off < BM; off += row_stride_a) {
        int row = inner_row_a + off;
        if (block_row_start + row < M) {
          for (int kk = 0; kk < 4; kk++) {
            int kidx = next_bk + inner_col_a * 4 + kk;
            float* dst = &As_nxt[row * (BK + 1) + inner_col_a * 4 + kk];
            if (kidx < K) {
              const float* src = &A_base[row * K + kidx];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
              __pipeline_memcpy_async(dst, src, sizeof(float));
#else
              *dst = *src;
#endif
            } else {
              *dst = 0.0f;
            }
          }
        } else {
          for (int kk = 0; kk < 4; kk++) {
            As_nxt[row * (BK + 1) + inner_col_a * 4 + kk] = 0.0f;
          }
        }
      }

      // Async load next B tile into Bs_nxt
      // Bs stride = BN = 128, which is 16-byte aligned, so float4 cp.async is safe
      #pragma unroll
      for (int off = 0; off < BK; off += row_stride_b) {
        int row = inner_row_b + off;
        int b_row = next_bk + row;
        if (b_row < K) {
          for (int nn = 0; nn < 4; nn++) {
            int nidx = inner_col_b * 4 + nn;
            float* dst = &Bs_nxt[row * BN + inner_col_b * 4 + nn];
            if (block_col_start + nidx < N) {
              const float* src = &B_base[b_row * N + nidx];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
              __pipeline_memcpy_async(dst, src, sizeof(float));
#else
              *dst = *src;
#endif
            } else {
              *dst = 0.0f;
            }
          }
        } else {
          for (int nn = 0; nn < 4; nn++) {
            Bs_nxt[row * BN + inner_col_b * 4 + nn] = 0.0f;
          }
        }
      }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
      __pipeline_commit();
#endif
    }

    // Compute current tile from As_cur / Bs_cur
    #pragma unroll
    for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
      #pragma unroll
      for (int wsri = 0; wsri < WMITER; wsri++) {
        #pragma unroll
        for (int i = 0; i < TM; i++) {
          reg_m[wsri * TM + i] =
              As_cur[(warp_row * WM + wsri * WSUBM +
                      thread_row_in_warp * TM + i) * (BK + 1) + dot_idx];
        }
      }
      #pragma unroll
      for (int wsci = 0; wsci < WNITER; wsci++) {
        #pragma unroll
        for (int i = 0; i < TN; i++) {
          reg_n[wsci * TN + i] =
              Bs_cur[dot_idx * BN + warp_col * WN +
                     wsci * WSUBN + thread_col_in_warp * TN + i];
        }
      }
      #pragma unroll
      for (int wsri = 0; wsri < WMITER; wsri++) {
        #pragma unroll
        for (int wsci = 0; wsci < WNITER; wsci++) {
          #pragma unroll
          for (int rm = 0; rm < TM; rm++) {
            float a_val = reg_m[wsri * TM + rm];
            #pragma unroll
            for (int rn = 0; rn < TN; rn++) {
              thread_results[(wsri * TM + rm) * (WNITER * TN) +
                             wsci * TN + rn] +=
                  a_val * reg_n[wsci * TN + rn];
            }
          }
        }
      }
    }

    if (has_next) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
      __pipeline_wait_prior(0);
#endif
      __syncthreads();

      // Swap buffers
      float* tmp;
      tmp = As_cur; As_cur = As_nxt; As_nxt = tmp;
      tmp = Bs_cur; Bs_cur = Bs_nxt; Bs_nxt = tmp;
    } else {
      // Last iteration: no next tile, just sync
      __syncthreads();
    }
  }

  // Write results (same as original kernel)
  #pragma unroll
  for (int wsri = 0; wsri < WMITER; wsri++) {
    #pragma unroll
    for (int wsci = 0; wsci < WNITER; wsci++) {
      float* C_sub = C + (wsri * WSUBM) * N + wsci * WSUBN;
      int gbase_m = block_row_start + warp_row * WM + wsri * WSUBM + thread_row_in_warp * TM;
      int gbase_n = block_col_start + warp_col * WN + wsci * WSUBN + thread_col_in_warp * TN;

      for (int rm = 0; rm < TM; rm++) {
        int gm = gbase_m + rm;
        if (gm >= M) continue;
        int gn = gbase_n;
        int idx = (wsri * TM + rm) * (WNITER * TN) + wsci * TN;
        float* c_addr = &C_sub[(thread_row_in_warp * TM + rm) * N +
                                thread_col_in_warp * TN];
        bool c_aligned = (((uintptr_t)c_addr) & 15) == 0;
        if (c_aligned && gn + TN <= N) {
          float4 val;
          val.x = thread_results[idx + 0];
          val.y = thread_results[idx + 1];
          val.z = thread_results[idx + 2];
          val.w = thread_results[idx + 3];
          reinterpret_cast<float4*>(c_addr)[0] = val;
        } else {
          for (int rn = 0; rn < TN; rn++) {
            if (gn + rn < N) {
              c_addr[rn] = thread_results[idx + rn];
            }
          }
        }
      }
    }
  }
}

#endif // MATMUL_KERNEL_CUH