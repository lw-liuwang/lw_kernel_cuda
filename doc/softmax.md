# Softmax

## 算子介绍

Softmax 将实数向量归一化为概率分布：
```
softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

广泛用于分类网络的最后一层和 Attention 机制。

## 算法原理

实现需要三个步骤：
1. 求最大值 `max(x)`（数值稳定性）
2. 计算 `exp(x_i - max)` 并求和
3. 归一化：每个元素除以总和

## 优化演进

| 版本 | 优化 | 说明 |
|------|------|------|
| kernel1 | 单线程处理一行 | 简单但线程利用率低 |
| kernel2 | shared memory 两阶段归约 | 减少全局内存访问 |
| kernel3 | warp shuffle 单 warp 归约 | 128 列以内无需 shmem |
| kernel4 | warp + shmem 多 warp 归约 | 任意列数都支持 |

## Profiling 分析

Softmax 是 **Memory-Bound** 算子（计算强度 < 1 FLOP/byte），瓶颈在显存带宽。

## 使用方式

```python
import torch
import lw_kernel_cuda

x = torch.randn(32, 4096, device='cuda', dtype=torch.float32)
out = lw_kernel_cuda.softmax(x)
```