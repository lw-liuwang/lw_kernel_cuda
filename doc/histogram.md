# Histogram: 直方图

## 算子介绍

将输入数据按值域分配到离散的桶中，统计每个桶中的元素数量。

## 算法原理

对每个元素，计算其所属的 bin 索引，然后对该 bin 计数器加 1。

## 优化演进

| 版本 | 优化 | 说明 |
|------|------|------|
| naive | 全局 atomicAdd | 大量冲突，性能差 |
| shared | 先写入共享内存，再归约到全局 | 减少全局原子操作冲突，性能大幅提升 |
| multi-block | 多 block 独立直方图加合并 | 适用于海量数据 |

## 使用方式

```python
import torch
import lw_kernel_cuda

x = torch.randn(10000, device='cuda', dtype=torch.float32)
bins = lw_kernel_cuda.histogram(x, num_bins=10, min_val=-3.0, max_val=3.0)
```