"""Benchmark Reduce operator vs torch.sum."""

from __future__ import annotations

import torch

import lw_kernel_cuda

from . import bench_utils as bu


def bench_reduce() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    Ns = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    for n in Ns:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        # Our implementation
        our_fn = lambda x=x: lw_kernel_cuda.reduce(x)
        our_ms, our_std = bu.measure_gpu_time(our_fn)

        # Reference: torch.sum
        ref_fn = lambda x=x: torch.sum(x)
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn)

        bytes_processed = n * 4  # read x (4 bytes per element)
        bw = bu.compute_bandwidth_gbps(bytes_processed, our_ms)
        ref_bw = bu.compute_bandwidth_gbps(bytes_processed, ref_ms)

        results.append(bu.BenchmarkResult(
            op_name="reduce",
            shape_desc=f"N={n}",
            our_ms=our_ms,
            our_std=our_std,
            ref_ms=ref_ms,
            ref_std=ref_std,
            metric=bw,
            ref_metric=ref_bw,
            metric_name="GB/s",
            bytes_processed=bytes_processed,
        ))

    return results


if __name__ == "__main__":
    bu.run_benchmarks(bench_reduce)