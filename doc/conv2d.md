# Conv2D: 二维卷积

## 算子介绍

二维卷积操作，输入形状为 (N, C, H, W)，卷积核形状为 (OC, IC, KH, KW)，输出形状为 (N, OC, OH, OW)。

## 算法原理

采用 **im2col + GEMM** 方法：
1. **im2col**: 将输入展开为矩阵 (N×OH×OW, C×KH×KW)
2. **GEMM**: 用矩阵乘法计算卷积

## 优化演进

| 版本 | 优化 | 说明 |
|------|------|------|
| naive | 直接 7 重循环 | 极度低效 |
| im2col+GEMM | 转换为矩阵乘法 | 利用高效的 MatMul 实现 |
| memory | 共享内存优化 im2col | 减少全局内存带宽压力 |

## 使用方式

```python
import torch
import lw_kernel_cuda

x = torch.randn(1, 3, 8, 8, device='cuda', dtype=torch.float32)
w = torch.randn(6, 3, 3, 3, device='cuda', dtype=torch.float32)
b = torch.randn(6, device='cuda', dtype=torch.float32)
out = lw_kernel_cuda.conv2d(x, w, b, stride=1, padding=0)
```