<!--
  lw-kernel-cuda: 高性能 CUDA 算子库

  基于 PyTorch Custom CUDA Extension 的高性能算子库，
  覆盖 10 个深度学习推理核心算子，采用 warp tiling、双缓冲流水线、warp 级归约等先进优化技术。
-->

<p align="center">
  <h1 align="center">lw-kernel-cuda</h1>
  <p align="center">高性能 CUDA 算子库</p>
  <p align="center">
    <a href="./BENCHMARKS.md">性能指标</a>
    ·
    <a href="./CHANGELOG.md">更新日志</a>
  </p>
</p>

## 简介

**lw-kernel-cuda** 是一个基于 PyTorch Custom CUDA Extension 的高性能算子库，覆盖深度学习推理中 10 个核心算子。

涵盖 **访存密集型**（VecAdd、Reduce、PrefixSum 等）和 **计算密集型**（MatMul、Conv2D、FlashAttention 等）两类场景，采用 warp tiling、双缓冲流水线、warp 级归约等 CUDA 优化技术，部分算子性能显著优于 PyTorch 参考实现。

### 性能亮点

| 算子 | 最佳性能 | vs PyTorch | 峰值利用率 |
|------|---------|-----------|-----------|
| reduce | **500 GB/s** | 持平 | 84% |
| vec_add | **473 GB/s** | 持平 | 79% |
| transpose | **411 GB/s** | **1.55x** | 69% |
| rmsnorm | **316 GB/s** | **2.75x** | 53% |
| histogram | **122 GB/s** | **74x** | 20% |
| matmul | **9.78 TFLOPS** | 80% | 31% |

> 完整性能数据参见 [BENCHMARKS.md](./BENCHMARKS.md)。测试环境：NVIDIA A10（峰值 600 GB/s, 31.2 TFLOPS FP32）。

## 算子列表

| 类别 | # | 算子 | 核心技术 | 优化亮点 |
|------|---|------|---------|---------|
| 访存密集型 | 1 | **VecAdd** | grid-stride loop, 合并访存 | 79% 带宽利用率 |
| | 2 | **Reduce** | shared memory, warp shuffle | 84% 带宽利用率 |
| | 3 | **Softmax** | online softmax, float4 向量化 | 融合 exp + sum 消除全局往返 |
| | 4 | **RMSNorm** | blockReduceSum, float4, rsqrtf | **2.75x vs PyTorch** |
| | 5 | **Transpose** | shared memory tiling, bank conflict 消除 | **1.55x vs PyTorch**, 含 padding 优化 |
| | 6 | **Histogram** | atomicAdd, 共享内存合并 | **74x vs PyTorch** (避免 GPU→CPU 传输) |
| | 7 | **PrefixSum** | Brent-Kung 算法, 多 block 扫描 | 修复多 block 正确性，支持任意规模 |
| 计算密集型 | 8 | **MatMul** | warp tiling, 寄存器 tiling, 双缓冲 + cp.async | 9.78 TFLOPS (31% 峰值) |
| | 9 | **Conv2D** | im2col + GEMM, 权重转置 | 随 MatMul 自动受益 |
| | 10 | **FlashAttention** | Br=4 tile, warp 级归约, float4 | **0.83 TFLOPS (7x vs 旧版)**, d=128 优化 |

## 优化技术体系

| 技术 | 说明 | 应用算子 |
|------|------|---------|
| **Warp tiling + 寄存器 tiling** | 分层 tile 分解，最大化寄存器复用 | MatMul, Conv2D |
| **双缓冲 + cp.async** | ping-pong 共享内存，隐藏全局加载延迟 | MatMul (db 变体) |
| **Warp 级归约** | 纯 shuffle 归约，消除 smem 与 sync 开销 | Reduce, Softmax, **FlashAttention** |
| **float4 向量化访存** | 128-bit 合并访存，提升带宽利用率 | VecAdd, Reduce, Softmax, RMSNorm, MatMul, FlashAttention |
| **Online softmax** | 单次前向 fused max + sum + output | Softmax, FlashAttention |
| **Grid-stride loop** | 自适应任意规模，简化边界处理 | VecAdd, Histogram, PrefixSum |
| **Bank conflict 消除** | 共享内存 padding 策略 | Transpose, MatMul |
| **Brent-Kung 扫描** | 高效 work-efficient 前缀和 | PrefixSum |

## 快速开始

### 环境要求

- CUDA Toolkit ≥ 12.0（推荐 12.4+）
- PyTorch ≥ 2.0
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
│   ├── matmul_kernel.cuh           # MatMul 模板（同步/双缓冲双版本）
│   └── cuda_helpers.h              # CUDA_CHECK 等辅助宏
├── doc/                            # 算子技术文档
├── benchmarks/                     # 性能测试框架
│   ├── run_all.py                  # 一键运行所有 benchmark
│   ├── bench_utils.py              # 基准测试工具
│   └── bench_*.py                  # 各算子独立 benchmark
├── lw_kernel_cuda/
│   ├── __init__.py                 # 导入 _C + 导出所有算子
│   ├── csrc/                       # C++/CUDA 源码
│   │   ├── extension.cpp           # TORCH_LIBRARY schema 注册
│   │   └── cuda/                   # 10 个算子的 .cu 实现
│   ├── ops/                        # Python 封装层
│   └── tests/                      # pytest 正确性测试
├── CHANGELOG.md                    # 版本更新记录
├── BENCHMARKS.md                   # 性能指标与分析
└── README.md                       # 本文件
```

## 质量保障

- **pytest 测试套件**：46 项测试覆盖 10 个算子，含边界条件（空张量、单元素、非对齐等）
- **PyTorch 对照验证**：每个算子的输出与 PyTorch 参考实现进行数值对校
- **统一 Benchmark 框架**：10 次 warmup + 100 次测量取平均，自动计算带宽 / TFLOPS

## 相关资源

- [cuBLAS](https://docs.nvidia.com/cuda/cublas/) / [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA 官方高性能线性代数库
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) — tiled online softmax 注意力机制
- [PyTorch Custom C++ & CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)