# Prefix Sum: 前缀和

## 算子介绍

前缀和（扫描）：`out[i] = sum_{j=0}^{i} in[j]`。看似简单的串行操作（O(N)），但并行化需要巧妙的设计。

## 算法原理

并行扫描的核心思路是使用树形结构，通过两阶段实现：
1. **Up-sweep (reduce)**: 构建归约树
2. **Down-sweep**: 从上到下分发前缀和

## 实现方法

| 算法 | 复杂度 | 特点 |
|------|--------|------|
| Kogge-Stone | O(N log N) 操作, O(log N) 步 | 适合小规模（<1024） |
| Brent-Kung | O(N log N) 操作, O(2 log N) 步 | 更少同步，适合大规模 |

## 使用方式

```python
import torch
import lw_kernel_cuda

x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda', dtype=torch.float32)
out = lw_kernel_cuda.prefix_sum(x)
# out = [1, 3, 6, 10]
```