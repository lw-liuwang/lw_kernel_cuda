"""Benchmark Histogram operator vs torch.histc."""

from __future__ import annotations

import torch

import lw_kernel_cuda

from . import bench_utils as bu


def bench_histogram() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    Ns = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    num_bins = 1024

    for n in Ns:
        x = torch.randn(n, device="cuda", dtype=torch.float32) * 3.0  # span [-3, 3]

        # Our implementation
        our_fn = lambda x=x: lw_kernel_cuda.histogram(x, num_bins, -3.0, 3.0)
        our_ms, our_std = bu.measure_gpu_time(our_fn)

        # Reference: torch.histc
        ref_fn = lambda x=x: torch.histc(x.cpu(), bins=num_bins, min=-3.0, max=3.0)
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn)

        bytes_processed = n * 4  # read x
        bw = bu.compute_bandwidth_gbps(bytes_processed, our_ms)
        ref_bw = bu.compute_bandwidth_gbps(bytes_processed, ref_ms)

        results.append(bu.BenchmarkResult(
            op_name="histogram",
            shape_desc=f"N={n},bins={num_bins}",
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
    bu.run_benchmarks(bench_histogram)