"""Benchmark RMSNorm operator vs torch.nn.functional.rms_norm."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import lw_kernel_cuda

from . import bench_utils as bu


def bench_rmsnorm() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    configs = [
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 8192),
        (16, 1024),
        (16, 2048),
        (16, 4096),
        (16, 8192),
        (128, 1024),
        (128, 2048),
        (128, 4096),
        (128, 8192),
        (512, 1024),
        (512, 2048),
        (512, 4096),
        (512, 8192),
    ]

    for rows, cols in configs:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        weight = torch.randn(cols, device="cuda", dtype=torch.float32)

        # Our implementation
        our_fn = lambda x=x, w=weight: lw_kernel_cuda.rmsnorm(x, w, 1e-6)
        our_ms, our_std = bu.measure_gpu_time(our_fn)

        # Reference: F.rms_norm
        ref_fn = lambda x=x, w=weight: F.rms_norm(x, (cols,), weight=w, eps=1e-6)
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn)

        bytes_processed = bu.rmsnorm_bytes(rows, cols)
        bw = bu.compute_bandwidth_gbps(bytes_processed, our_ms)
        ref_bw = bu.compute_bandwidth_gbps(bytes_processed, ref_ms)

        results.append(bu.BenchmarkResult(
            op_name="rmsnorm",
            shape_desc=f"{rows}x{cols}",
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
    bu.run_benchmarks(bench_rmsnorm)