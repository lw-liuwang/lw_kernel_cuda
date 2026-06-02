# FlashAttention

## 算子介绍

FlashAttention 通过 **tiling + online softmax** 技术，避免 Attention 计算中的大中间矩阵 (N×N)，将显存复杂度从 O(N²) 降到 O(N)。

## 算法原理

传统 Attention 计算:
```
S = Q @ K^T         # (N, N)
P = softmax(S)       # (N, N)
O = P @ V            # (N, d)
```

FlashAttention 将 Q、K、V 分块处理。对每块 K/V：
1. 计算局部 QK^T
2. 用 **online softmax** 更新：`new_max = max(old_max, local_max)`，根据 new_max 对旧输出缩放到新尺度
3. 累加局部加权和到输出

## Online Softmax 核心更新

```cuda
float rescale = expf(old_max - new_max);
float new_denom = old_denom * rescale + local_denom;
O = O * rescale + exp(S_local - new_max) @ V_local;
```

## 优化要点

- 使用共享内存 tile 存储 Q/K/V 块
- 避免全局内存中 N×N 的 attention 矩阵
- Kernel fusion：将 attention 的所有步骤合并到一个 kernel

## 使用方式

```python
import torch
import lw_kernel_cuda

B, H, seqlen, dim = 2, 4, 16, 128
q = torch.randn(B, H, seqlen, dim, device='cuda', dtype=torch.float32)
k = torch.randn(B, H, seqlen, dim, device='cuda', dtype=torch.float32)
v = torch.randn(B, H, seqlen, dim, device='cuda', dtype=torch.float32)
out = lw_kernel_cuda.flash_attention(q, k, v)
```

## 参考文献

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)