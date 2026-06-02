# Reduce: 归约求和

## 算子介绍

将输入数组中的所有元素累加得到一个标量值。是许多算子的基础组件（如 softmax、rmsnorm 中的求和步骤）。

## 算法原理

归约的核心是将 N 个元素通过某种结合律操作（如加法）规约为 1 个值。串行复杂度为 O(N)，但并行化需要树形规约，复杂度为 O(log N)。

## 实现

分两阶段：
1. **第一阶段**: 每个 block 对部分数据进行 warp shuffle + shared memory 归约，输出部分和
2. **第二阶段**: 一个 block 归约所有部分和到最终结果

核心使用 `blockReduceSum`：
```cuda
float sum = 0.0f;
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
     i += gridDim.x * blockDim.x) {
  sum += in[i];
}
sum = blockReduceSum(sum);
```

## 优化演进

| 版本 | 优化 | 加速比 |
|------|------|--------|
| v0 (naive) | 单线程单元素 | 1x |
| v1 | 全局内存原子操作 | 2x |
| v2 | 共享内存树形归约 | 5x |
| v3 | Warp shuffle 减少同步 | 8x |
| v4 | 完整 warp + shmem | 10x |

## 参考文献

- [CUDA Reduction 官方博客](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)