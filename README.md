# lw-kernel-cuda

从零搭建的 CUDA 算子库，使用 PyTorch Custom CUDA Extension 实现，覆盖深度学习推理中 10 个核心算子。

## 算子列表（从易到难）

| # | 算子 | 核心知识点 | 难度 |
|---|------|-----------|------|
| 1 | **[VecAdd](doc/vec_add.md)** | grid-stride loop, CUDA 线程模型, 显存合并访问 | ⭐ |
| 2 | **[Reduce](doc/reduce.md)** | shared memory, warp shuffle (`__shfl_down_sync`), block reduce 模式 | ⭐⭐ |
| 3 | **[Softmax](doc/softmax.md)** | 多阶段 reduce (max + sum), warp + shared memory 组合, online softmax | ⭐⭐ |
| 4 | **[RMSNorm](doc/rmsnorm.md)** | blockReduceSum, float4 向量化访存, rsqrtf 归一化 | ⭐⭐ |
| 5 | **[Transpose](doc/transpose.md)** | shared memory tiling, bank conflict 与 padding 优化, 合并与非合并访存 | ⭐⭐⭐ |
| 6 | **[Histogram](doc/histogram.md)** | atomicAdd, 共享内存直方图合并, 原子操作性能分析 | ⭐⭐⭐ |
| 7 | **[PrefixSum](doc/prefix_sum.md)** | Kogge-Stone 算法, Brent-Kung 算法, 多 block 扫描, 并行前缀和 | ⭐⭐⭐ |
| 8 | **[MatMul](doc/matmul.md)** | 共享内存 tiling, register tiling, 双缓冲, warp tiling, compute-bound 优化 | ⭐⭐⭐⭐ |
| 9 | **[Conv2D](doc/conv2d.md)** | im2col + GEMM 组合, 卷积的 im2col 加速, 权重转置 | ⭐⭐⭐⭐ |
| 10 | **[FlashAttention](doc/flash_attention.md)** | tiled online softmax, kernel fusion, 显存复杂度 O(N²)→O(N), 分块计算 | ⭐⭐⭐⭐⭐ |

## 快速开始

### 构建

```bash
cd lw-kernel-cuda
python3 setup.py build_ext --inplace
```

### 运行测试

```bash
python3 -m pytest lw_kernel_cuda/tests/ -v
```

所有 45 个测试通过（覆盖 10 个算子）。

### 使用示例

```python
import lw_kernel_cuda
import torch

# VecAdd
a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
c = lw_kernel_cuda.vec_add(a, b)

# MatMul
a = torch.randn(256, 128, device="cuda")
b = torch.randn(128, 256, device="cuda")
c = lw_kernel_cuda.matmul(a, b)
```

## 项目结构

```
lw-kernel-cuda/
├── setup.py                        # PyTorch extension 构建
├── include/                        # 共享工具头文件
│   ├── cuda_helpers.h              # CUDA_CHECK 宏
│   ├── warp_reduce.h               # warpReduceSum / warpReduceMax
│   ├── block_reduce.h              # blockReduceSum / blockReduceMax
│   └── block_softmax.h             # online-softmax 辅助函数
├── doc/                            # 算子文档（10 篇）
├── lw_kernel_cuda/
│   ├── __init__.py                 # 导入 _C + 导出所有算子
│   ├── csrc/
│   │   ├── extension.cpp           # TORCH_LIBRARY schema 注册
│   │   └── cuda/                   # 10 个算子的 CUDA kernel
│   ├── ops/                        # Python 封装（10 个）
│   └── tests/                      # 正确性测试（10 个）
```

## 关键技术点

| 技术 | 涉及的算子 |
|------|-----------|
| Grid-stride loop | VecAdd, Histogram |
| Shared memory | Reduce, Softmax, Transpose, MatMul, Conv2D, FlashAttention |
| Warp shuffle | Reduce, Softmax |
| Vectorized memory access (float4) | RMSNorm |
| Bank conflict avoidance | Transpose |
| Atomic operations | Histogram |
| Register tiling | MatMul, Conv2D |
| Double buffering | MatMul |
| Online softmax | Softmax, FlashAttention |
| Kernel fusion | FlashAttention |

## 性能指标

测试环境: NVIDIA A10 (峰值 600 GB/s, 31.2 TFLOPS FP32), CUDA 13.0, PyTorch 2.6, 100 次测量平均。

### 访存密集型算子

| 算子 | 测试规模 | 本实现 | PyTorch 参考 | 加速比 | 峰值利用率 |
|------|---------|--------|-------------|--------|-----------|
| vec_add | N=100M | 474 GB/s | 481 GB/s | 0.98x | 79% |
| reduce | N=100M | 501 GB/s | 506 GB/s | 0.99x | 84% |
| rmsnorm | 512x4096 | 314 GB/s | 115 GB/s | **2.74x** | 52% |
| transpose | 4096x4096 | 412 GB/s | 264 GB/s | **1.56x** | 69% |
| histogram | N=10M | 122 GB/s | 2 GB/s | **70x** | 20% |
| prefix_sum | N=100M | 143 GB/s | 481 GB/s | 0.30x | 24% |

### 计算密集型算子

| 算子 | 测试规模 | 本实现 | PyTorch 参考 | 加速比 | 峰值利用率 |
|------|---------|--------|-------------|--------|-----------|
| softmax | 512x2048 | 336 GB/s | 289 GB/s | **1.16x** | 56% |
| softmax | 2048x4096 | 227 GB/s | 462 GB/s | 0.49x | 38% |
| matmul | 2048x2048x2048 | 9.31 TFLOPS | 15.5 TFLOPS | 0.60x | 30% |
| conv2d | 4x3x224x224_64x3x7x7 | 1.84 TFLOPS | 8.14 TFLOPS | 0.23x | 6% |
| flash_attn | B1H1S512D128 | 0.12 TFLOPS | 1.31 TFLOPS | 0.09x | 0.4% |

### 关键发现

- **rmsnorm**, **transpose**, **histogram** 显著优于 PyTorch 参考实现（2.7x ~ 70x），得益于 float4 向量化和优化的共享内存策略
- **vec_add**, **reduce** 接近设备峰值带宽（79-84%），与 PyTorch 性能相当
- **softmax** 融合了 exp + sum 消除全局内存往返，中等规模行（512x2048）超越参考，但超大行（2048x4096）受限于单步处理策略，尚有优化空间
- **matmul** 采用 warp tiling + float4 + bank conflict 消除，达 9.3 TFLOPS（30% 峰值），仍有提升空间（如双缓冲、double MMA pipeline）
- **prefix_sum** 多块 Brent-Kung 算法正确性已修复，但三步法 kernel launch 开销大，大 N 性能不理想（可使用 CUB 替代）
- **flash_attention** 是当前最大短板，Br=1 设计导致 O(S²) 次 blockReduceSum 调用（每个 K/V 位置一次）。提升需增加 Br tile 大小（Br > 1, 多行 Q 联合处理，分摊 K/V 加载代价）
- **conv2d** 受限于 im2col + GEMM 路线本身（显存开销大），且继承了 matmul 的性能上限

### 优化技术覆盖

| 优化技术 | 算子 |
|---------|------|
| float4 向量化访存 | vec_add, reduce, softmax, rmsnorm, matmul |
| 融合 kernel | softmax (exp+sum 融合), flash_attention (online softmax + output) |
| blockReduceSum/Max | softmax, rmsnorm, flash_attention |
| warp shuffle | reduce, softmax |
| 共享内存 padding | transpose, matmul (As 的 +1 padding) |
| warp tiling + register tiling | matmul, conv2d |
| warp-group privatization | histogram (2-segment 共享内存原子操作) |
| Brent-Kung work-efficient scan | prefix_sum |