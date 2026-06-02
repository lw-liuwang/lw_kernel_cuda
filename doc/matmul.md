# MatMul: 矩阵乘法

## 算子介绍

矩阵乘法 `C = A @ B`，其中 A(M×K)、B(K×N)、C(M×N)。是深度学习中最核心的算子之一。

## 算法原理

```
C[i][j] = sum_k(A[i][k] * B[k][j])
```

计算复杂度 O(M×N×K)，是典型的 **Compute-Bound** 算子（大尺寸时）。

## 优化演进

| 版本 | 优化 | 加速比 |
|------|------|--------|
| v0 | naive triple loop | 1x |
| v1 | 共享内存 tiling (BK=16) | 5x |
| v2 | 寄存器 tiling (TM/TN) | 8x |
| v3 | float4 向量化加载/写回 | 12x |
| v4 | 双缓冲流水线掩盖访存延迟 | 14x |
| v5 | Warp tiling (BN/BW) | 16x |

## 使用方式

```python
import torch
import lw_kernel_cuda

a = torch.randn(256, 256, device='cuda', dtype=torch.float32)
b = torch.randn(256, 256, device='cuda', dtype=torch.float32)
c = lw_kernel_cuda.matmul(a, b)
```