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

#endif // MATMUL_KERNEL_CUH