# Transpose

## 算子介绍

矩阵转置：`B[j][i] = A[i][j]`。看似简单，但非合并访存导致性能下降严重。

## 算法原理

Naive 转置的问题是全局内存访问模式不连续。A 的行读是连续的（合并访问），但 B 的列写是不连续的（非合并访问）。

## 优化演进

| 版本 | 优化 | 说明 |
|------|------|------|
| naive | 直接全局内存读写 | 写端非合并访存 |
| smem tiling | 共享内存做中转 | 合并读 + 合并写 |
| padding | `[32][32+1]` 避免 bank conflict | 消除 bank conflict |
| unroll | 循环展开提升 ILP | 提高指令级并行 |

## Bank Conflict

共享内存有 32 个 bank。当 `tile[TILE][TILE]` 的 `TILE=32` 时，同一列元素映射到同一 bank，发生 32 路 bank conflict。padding 为 `[32][33]` 后，不再有 bank conflict。

## 使用方式

```python
import torch
import lw_kernel_cuda

x = torch.randn(256, 512, device='cuda', dtype=torch.float32)
out = lw_kernel_cuda.transpose(x)
```