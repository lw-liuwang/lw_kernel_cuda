"""Benchmark MatMul operator vs a@b (cuBLAS). Reports TFLOPS."""

from __future__ import annotations

import torch

import lw_kernel_cuda

from . import bench_utils as bu


def bench_matmul() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    configs = [
        # (M, N, K)  square
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        # rectangular: M=N, K=small
        (1024, 1024, 64),
        (1024, 1024, 256),
        (4096, 4096, 64),
        (4096, 4096, 256),
        # rectangular: M=large, N=small
        (4096, 512, 1024),
        (512, 4096, 1024),
        # MLP-like
        (16384, 1024, 4096),
        (16384, 4096, 1024),
    ]

    for M, N, K in configs:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Our implementation
        our_fn = lambda a=a, b=b: lw_kernel_cuda.matmul(a, b)
        our_ms, our_std = bu.measure_gpu_time(our_fn)

        # Reference: cuBLAS via a @ b
        ref_fn = lambda a=a, b=b: a @ b
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn)

        flops = bu.matmul_flops(M, N, K)
        tflops = bu.compute_tflops(flops, our_ms)
        ref_tflops = bu.compute_tflops(flops, ref_ms)
        bytes_ = bu.matmul_bytes(M, N, K)

        results.append(bu.BenchmarkResult(
            op_name="matmul",
            shape_desc=f"{M}x{N}x{K}",
            our_ms=our_ms,
            our_std=our_std,
            ref_ms=ref_ms,
            ref_std=ref_std,
            metric=tflops,
            ref_metric=ref_tflops,
            metric_name="TFLOPS",
            bytes_processed=bytes_,
            flops=flops,
        ))

    return results


if __name__ == "__main__":
    bu.run_benchmarks(bench_matmul)