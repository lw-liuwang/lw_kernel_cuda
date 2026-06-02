# VecAdd: 向量加法

## 算子介绍

逐元素向量加法，是最简单的 CUDA 入门算子。计算 `C[i] = A[i] + B[i]`，其中 A、B、C 是长度相同的浮点向量。

## 算法原理

每个 CUDA 线程负责处理一个或多个元素，采用 **grid-stride loop** 模式确保任意长度的向量都能被处理。

## 实现

每个线程使用 grid-stride loop 遍历向量：
```cuda
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
     i += gridDim.x * blockDim.x) {
  c[i] = a[i] + b[i];
}
```

## 优化演进

| 版本 | 优化 | 说明 |
|------|------|------|
| naive | 1 thread = 1 element | 简单但无法处理超大向量 |
| grid-stride | 自动跨步循环 | 任意长度向量都支持 |

## Profiling 分析

VecAdd 是典型的 **Memory-Bound** 算子：
- 计算强度 ≈ 1 FLOP / 12 bytes (3 × float32)
- 瓶颈在显存带宽，优化方向是通过向量化访存提升带宽利用率

## 使用方式

```python
import torch
import lw_kernel_cuda

a = torch.randn(10000, device='cuda', dtype=torch.float32)
b = torch.randn(10000, device='cuda', dtype=torch.float32)
c = lw_kernel_cuda.vec_add(a, b)
```