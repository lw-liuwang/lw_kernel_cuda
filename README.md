<!--
  lw-kernel-cuda: 从零搭建的 CUDA 算子库

  一个专注于深度学习推理场景、从零手写的 CUDA 算子集合。
  覆盖 10 个核心算子，从简单到复杂逐步优化，每个算子在 doc/ 下有对应的原理讲解文档。
-->

<p align="center">
  <h1 align="center">lw-kernel-cuda</h1>
  <p align="center">从零搭建的深度学习 CUDA 算子库</p>
  <p align="center">
    <a href="./BENCHMARKS.md">性能指标</a>
    ·
    <a href="./CHANGELOG.md">更新日志</a>
  </p>
</p>

## 简介

**lw-kernel-cuda** 是一个从零手写的 CUDA 算子库，基于 PyTorch Custom CUDA Extension 实现。

覆盖深度学习推理中的 10 个核心算子，涵盖 **访存密集型**（VecAdd、Reduce、PrefixSum 等）和 **计算密集型**（MatMul、Conv2D、FlashAttention 等）两类场景。

目标：在动手实现每个算子的过程中，系统性地掌握 CUDA 优化方法论。

## 算子列表

| # | 算子 | 核心知识点 | 难度 |
|---|------|-----------|------|
| 1 | **[VecAdd](doc/vec_add.md)** | grid-stride loop, 线程模型, 合并访存 | ⭐ |
| 2 | **[Reduce](doc/reduce.md)** | shared memory, warp shuffle, block reduce 模式 | ⭐⭐ |
| 3 | **[Softmax](doc/softmax.md)** | 多阶段 reduce (max + sum), online softmax | ⭐⭐ |
| 4 | **[RMSNorm](doc/rmsnorm.md)** | blockReduceSum, float4 向量化, rsqrtf | ⭐⭐ |
| 5 | **[Transpose](doc/transpose.md)** | shared memory tiling, bank conflict 与 padding, 合并/非合并访存 | ⭐⭐⭐ |
| 6 | **[Histogram](doc/histogram.md)** | atomicAdd, 共享内存直方图合并, 原子操作性能 | ⭐⭐⭐ |
| 7 | **[PrefixSum](doc/prefix_sum.md)** | Kogge-Stone / Brent-Kung 算法, 多 block 扫描 | ⭐⭐⭐ |
| 8 | **[MatMul](doc/matmul.md)** | 共享内存 tiling, warp tiling, 寄存器 tiling, 双缓冲 | ⭐⭐⭐⭐ |
| 9 | **[Conv2D](doc/conv2d.md)** | im2col + GEMM, 卷积加速, 权重转置 | ⭐⭐⭐⭐ |
| 10 | **[FlashAttention](doc/flash_attention.md)** | tiled online softmax, kernel fusion, O(N²)→O(N) 显存 | ⭐⭐⭐⭐⭐ |

## 快速开始

### 环境要求

- CUDA Toolkit ≥ 12.0（推荐 12.4+）
- PyTorch ≥ 2.0
- Python ≥ 3.10
- GPU: 计算能力 8.0+（Ampere 架构及以上，用于 cp.async 等特性）

### 构建

```bash
cd lw-kernel-cuda
python3 setup.py build_ext --inplace
```

### 测试

```bash
python3 -m pytest lw_kernel_cuda/tests/ -v
```

所有测试通过后，即可开始使用。

### 使用示例

```python
import lw_kernel_cuda
import torch

# 向量加法
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = lw_kernel_cuda.vec_add(a, b)

# 矩阵乘法
a = torch.randn(256, 128, device="cuda")
b = torch.randn(128, 256, device="cuda")
c = lw_kernel_cuda.matmul(a, b)

# Softmax
x = torch.randn(32, 4096, device="cuda")
y = lw_kernel_cuda.softmax(x)

# Flash Attention
q = torch.randn(1, 4, 512, 128, device="cuda")
k = torch.randn(1, 4, 512, 128, device="cuda")
v = torch.randn(1, 4, 512, 128, device="cuda")
o = lw_kernel_cuda.flash_attention(q, k, v)
```

## 项目结构

```
lw-kernel-cuda/
├── setup.py                        # PyTorch extension 构建脚本
├── include/                        # 共享工具头文件
│   ├── block_reduce.h              # blockReduceSum / blockReduceMax
│   ├── matmul_kernel.cuh           # MatMul 模板（含同步/双缓冲双版本）
│   └── cuda_helpers.h              # CUDA_CHECK 等辅助宏
├── doc/                            # 10 个算子的原理与优化文档
│   ├── vec_add.md
│   ├── flash_attention.md
│   └── ...
├── benchmarks/                     # 性能测试（统一框架）
│   ├── run_all.py                  # 一键运行所有 benchmark
│   ├── bench_utils.py              # 基准测试工具函数
│   └── bench_*.py                  # 每个算子的独立 benchmark
├── lw_kernel_cuda/
│   ├── __init__.py                 # 导入 _C + 导出所有算子
│   ├── _C.cpython-*.so             # 编译后的扩展（运行时生成）
│   ├── csrc/                       # C++/CUDA 源码
│   │   ├── extension.cpp           # TORCH_LIBRARY schema 注册
│   │   └── cuda/                   # 10 个算子的 .cu 实现
│   ├── ops/                        # Python 封装层
│   └── tests/                      # 正确性测试（pytest）
├── CHANGELOG.md                    # 版本更新记录
├── BENCHMARKS.md                   # 性能指标与分析
└── README.md                       # 本文件
```

## 优化技术一览

| 技术 | 涉及的算子 |
|------|-----------|
| Grid-stride loop | VecAdd, Histogram, PrefixSum |
| Shared memory tiling | Reduce, Softmax, Transpose, MatMul, Conv2D, FlashAttention |
| Warp shuffle 归约 | Reduce, Softmax, **FlashAttention (Br=4 kernel)** |
| float4 向量化访存 | VecAdd, Reduce, Softmax, RMSNorm, MatMul, **FlashAttention** |
| 融合 kernel | Softmax (exp+sum), FlashAttention (online softmax + output) |
| Bank conflict 消除 | Transpose (padding), MatMul (+1 stride) |
| 原子操作 | Histogram |
| 寄存器 tiling | MatMul, Conv2D |
| **双缓冲 + cp.async** | **MatMul (db 变体)** |
| **Warp 级 reduction** | **FlashAttention (替代 blockReduce 消除 sync)** |
| Brent-Kung 算法 | PrefixSum |

## 设计原则

1. **从 naive 到优化**：每个算子提供多个优化版本的演进路径，对应课程文档的讲解节奏
2. **可读性优先**：kernel 代码注释完整，关键优化点有对应文档解释
3. **与 PyTorch 对照**：每个算子的测试和 benchmark 均以 PyTorch 参考实现为 baseline
4. **正确性保障**：基于 pytest 的测试体系，覆盖边界条件（空张量、单元素、非对齐等）

## 相关资源

- [cuBLAS](https://docs.nvidia.com/cuda/cublas/) / [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA 官方高性能线性代数库
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) — 本文 kernel fusion 思路的来源
- [PyTorch Custom C++ & CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)