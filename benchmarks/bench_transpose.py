"""Benchmark Transpose operator vs x.t().contiguous()."""

from __future__ import annotations

import torch

import lw_kernel_cuda

from . import bench_utils as bu


def bench_transpose() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    for n in sizes:
        x = torch.randn(n, n, device="cuda", dtype=torch.float32)

        # Our implementation
        our_fn = lambda x=x: lw_kernel_cuda.transpose(x)
        our_ms, our_std = bu.measure_gpu_time(our_fn)

        # Reference: x.t().contiguous()
        ref_fn = lambda x=x: x.t().contiguous()
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn)

        bytes_processed = 2 * n * n * 4  # read + write
        bw = bu.compute_bandwidth_gbps(bytes_processed, our_ms)
        ref_bw = bu.compute_bandwidth_gbps(bytes_processed, ref_ms)

        results.append(bu.BenchmarkResult(
            op_name="transpose",
            shape_desc=f"{n}x{n}",
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
    bu.run_benchmarks(bench_transpose)