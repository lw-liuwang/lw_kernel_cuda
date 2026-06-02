"""Benchmark VecAdd operator vs torch.add."""

from __future__ import annotations

import torch

import lw_kernel_cuda

from . import bench_utils as bu


def bench_vec_add() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    Ns = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    for n in Ns:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        # Our implementation
        our_fn = lambda a=a, b=b: lw_kernel_cuda.vec_add(a, b)
        our_ms, our_std = bu.measure_gpu_time(our_fn)

        # Reference: torch.add
        ref_fn = lambda a=a, b=b: torch.add(a, b)
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn)

        bytes_processed = 3 * n * 4  # read a, read b, write out
        bw = bu.compute_bandwidth_gbps(bytes_processed, our_ms)
        ref_bw = bu.compute_bandwidth_gbps(bytes_processed, ref_ms)

        results.append(bu.BenchmarkResult(
            op_name="vec_add",
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
    bu.run_benchmarks(bench_vec_add)