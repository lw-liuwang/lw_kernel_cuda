# Changelog

## 第三轮优化 (2026-06-02)

### 优化概览

对本轮 2 个算子进行了进一步优化，benchmark 测试通过，完整测试套件 46 项全部通过。

---

### P0: Flash Attention Br=4 → Br=8 全面升级

**文件**: `lw_kernel_cuda/csrc/cuda/flash_attention.cu`

- **新增 `flash_attention_br8_kernel`**（d=128, float4）和 **`flash_attention_br8_kernel_d64`**（d=64, float2）：
  - 每个 warp 处理 2 个 query row（Br=4 时每个 warp 处理 1 个）
  - grid 从 `(B*H, S/4)` 缩小到 `(B*H, S/8)`，K/V 全局读取进一步减半
  - 每线程维护 2 套 online softmax 状态和 2 套 accumulator
  - 每 K 位置计算 2 次 dot product + 2 次 warpReduceSum（纯 shuffle）
- **替换 Br=4 kernel** 成为 d=128/d=64 的主 kernel
- Bc=64 尝试：64KB 共享内存改变了 L1/shared 分区，d=128 上性能倒退，未采用
- Br=4 kernel 保留作为代码参考

**性能**: d=128 B1H1S4096 从 0.83 → **1.16 TFLOPS**（1.40x），d=64 从 0.67 → **0.93 TFLOPS**（1.39x）

---

### P0: Prefix Sum 每线程 2 元素 + 正确 Exclusive Scan

**文件**: `lw_kernel_cuda/csrc/cuda/prefix_sum.cu`

- **新增 `scan_blocks_kernel` 每线程 2 元素版本**：
  - 每个 block 处理 `2 * blockDim.x` 元素，grid size 减半
  - 使用 up-sweep + inclusive post-scan（Blelloch 算法）
- **修复 `scan_block_sums_kernel`**：
  - 改用正确的 **Blelloch exclusive down-sweep**（set-last-to-zero + copy-and-add）
  - 原实现使用了 inclusive post-scan（与 `scan_blocks_kernel` 相同），误用于需要 exclusive 结果的 block_sums 扫描
  - 新实现: `stride` 从 `chunk_size>>1` 开始，`right < chunk_size` 条件，`t=shared[left]; shared[left]=shared[right]; shared[right]+=t;`
  - 配合 grid-stride loop 的 carry 累加，正确计算出全局 exclusive prefix
- **更新 `add_block_sums_kernel`**：适配每线程 2 元素的 block 布局

**性能**: N=100M 从 107 → **171 GB/s**（~1.6x），正确性通过所有测试（含 N=100000 多 block 路径）

---

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