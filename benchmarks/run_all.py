#!/usr/bin/env python3
"""
Unified benchmark runner for lw-kernel-cuda.

Runs all 10 operator benchmarks sequentially and exports results to CSV.

Usage:
    python -m benchmarks.run_all
    python benchmarks/run_all.py          # same effect
"""

from __future__ import annotations

import os
import sys
import time

# Add project root to path so we can import lw_kernel_cuda
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from . import bench_utils as bu
from .bench_vec_add import bench_vec_add
from .bench_reduce import bench_reduce
from .bench_softmax import bench_softmax
from .bench_rmsnorm import bench_rmsnorm
from .bench_transpose import bench_transpose
from .bench_histogram import bench_histogram
from .bench_prefix_sum import bench_prefix_sum
from .bench_matmul import bench_matmul
from .bench_conv2d import bench_conv2d
from .bench_flash_attention import bench_flash_attention


def main():
    print("=" * 70)
    print("  lw-kernel-cuda Benchmark Suite")
    print(f"  Device: {bu.DEVICE_NAME}")
    print(f"  Theoretical Peak: {bu.MEMORY_BW_GBPS:.0f} GB/s, {bu.FP32_TFLOPS:.1f} TFLOPS")
    print(f"  Warmup: {bu.WARMUP_ITERS}, Measured: {bu.MEASURED_ITERS}")
    print("=" * 70)

    all_results: list[bu.BenchmarkResult] = []
    t_start = time.time()

    bench_fns = [
        ("vec_add", bench_vec_add),
        ("reduce", bench_reduce),
        ("softmax", bench_softmax),
        ("rmsnorm", bench_rmsnorm),
        ("transpose", bench_transpose),
        ("histogram", bench_histogram),
        ("prefix_sum", bench_prefix_sum),
        ("matmul", bench_matmul),
        ("conv2d", bench_conv2d),
        ("flash_attention", bench_flash_attention),
    ]

    for name, fn in bench_fns:
        try:
            results = bu.run_benchmarks(fn)
            all_results.extend(results)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")

    total_elapsed = time.time() - t_start

    # Export CSV
    csv_path = os.path.join(os.path.dirname(__file__), "results.csv")
    bu.export_csv(all_results, csv_path)

    print(f"\n{'=' * 70}")
    print(f"  All benchmarks completed in {total_elapsed:.1f}s")
    print(f"  Results exported to: {csv_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()