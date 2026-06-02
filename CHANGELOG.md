# Changelog

## 第二轮优化 (2026-06-02)

### 优化概览

对本轮 3 个算子进行了 P0/P1 级别的优化和修复，benchmark 测试通过，完整测试套件 46 项全部通过。

---

### P0: Flash Attention Br=4 + Warp-Level Reduction

**文件**: `lw_kernel_cuda/csrc/cuda/flash_attention.cu`

- **新增 `flash_attention_br4_kernel`**：面向 d=128 的 Br=4 优化 kernel
  - blockDim=128（4 个 warp），每个 warp 独立处理一个 query row
  - 每线程 float4（4 个 d 元素），向量化加载 Q/K/V
  - **warpReduceSum**（纯 shuffle，无 smem/sync）替代 blockReduceSum
  - grid 从 `(B*H, seqlen)` 缩小到 `(B*H, seqlen/4)`，全局 K/V 读取减少 4 倍
- **保留 `flash_attention_br1_kernel`**：作为 d=64 等小维度的 fallback

**性能**: d=128 从 0.12 → 0.83 TFLOPS（~7x），峰值利用率 2.7%

---

### P0: Prefix Sum 多 Block 扫描正确性修复

**文件**: `lw_kernel_cuda/csrc/cuda/prefix_sum.cu`、`lw_kernel_cuda/tests/test_prefix_sum.py`

- **修复 `scan_block_sums_kernel`**：
  - 原实现仅启动 1 个 block (blockDim=256)，无法处理 grid_size > 256 的情况
  - 当 N > 65536 时，block_sums 数组后半段未被扫描，导致后续加偏移时结果错误
  - 改为 grid-stride 循环：逐 chunk 进行 Brent-Kung 扫描，跨 chunk 维护 carry
- **新增测试**: `test_prefix_sum_large`（N=100000 > 65536），验证多 block 路径

**状态**: 正确性修复，大 N 性能仍有优化空间（107 GB/s，三步法 kernel launch 开销大）

---

### P1: MatMul 双缓冲 + cp.async

**文件**: `include/matmul_kernel.cuh`、`lw_kernel_cuda/csrc/cuda/matmul.cu`

- **新增 `matmul_kernel_db` template**：双缓冲变体
  - ping-pong 共享内存（`As_ping/Bs_ping`, `As_pong/Bs_pong`）
  - 首 tile 同步加载，后续 tile 使用 `__pipeline_memcpy_async`（4-byte cp.async）异步预取
  - 计算当前 tile 与预取下个 tile 流水线并行
- **默认启用**: `matmul_cuda` 改为调用 `matmul_kernel_db`，保留 `matmul_cuda_sync` 调用原 kernel

**性能**: 9.3 → 9.78 TFLOPS（~5%）。计算瓶颈（而非访存瓶颈）限制了 cp.async 的收益

---

### 文档更新

- 重写 `README.md`，优化项目介绍和结构说明
- 新建 `BENCHMARKS.md`，汇总所有算子的详细性能数据和分析

---

## 第一轮优化 (Phase 1-5)

初始版本，完成 10 个算子的 naive → 优化实现，45 个测试通过。

各算子的详细优化历程参见 `doc/` 目录下的对应文档。