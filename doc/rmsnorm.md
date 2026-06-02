# RMSNorm

## 算子介绍

Root Mean Square Layer Normalization (RMSNorm) 是 LayerNorm 的简化版本：
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```

相比 LayerNorm 去掉了均值中心化步骤，计算更高效，在 LLM 中广泛使用（如 LLaMA）。

## 算法原理

1. 计算均方根：`rms = sqrt(1/N * sum(x_i^2) + eps)`
2. 归一化并加权：`out_i = x_i / rms * weight_i`

## 优化要点

- **float4 向量化访存**: 一次加载 4 个 float，提升带宽利用率
- **blockReduceSum**: warp shuffle + shared memory 归约平方和
- **rsqrtf**: 使用 CUDA 内置快速倒数平方根指令

## 使用方式

```python
import torch
import lw_kernel_cuda

x = torch.randn(16, 4096, device='cuda', dtype=torch.float32)
w = torch.randn(4096, device='cuda', dtype=torch.float32)
out = lw_kernel_cuda.rmsnorm(x, w, eps=1e-6)
```

## 参考文献

- [RMSNorm: Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)