"""Benchmark FlashAttention operator vs F.scaled_dot_product_attention. Reports TFLOPS."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import lw_kernel_cuda

from . import bench_utils as bu


def bench_flash_attention() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    configs = [
        # d=64 configs
        (1, 1, 512, 64),
        (1, 1, 1024, 64),
        (1, 1, 2048, 64),
        (1, 1, 4096, 64),
        (1, 4, 512, 64),
        (1, 4, 1024, 64),
        (1, 4, 2048, 64),
        (1, 4, 4096, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (4, 4, 512, 64),
        (4, 4, 1024, 64),
        (4, 8, 512, 64),
        (4, 8, 1024, 64),
        # d=128 configs (our kernel is optimized for d=128)
        (1, 1, 512, 128),
        (1, 1, 1024, 128),
        (1, 1, 2048, 128),
        (1, 1, 4096, 128),
        (1, 4, 512, 128),
        (1, 4, 1024, 128),
        (1, 4, 2048, 128),
        (1, 8, 512, 128),
        (1, 8, 1024, 128),
        (4, 4, 512, 128),
        (4, 4, 1024, 128),
    ]

    for B, H, S, D in configs:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)

        # Our implementation
        our_fn = lambda q=q, k=k, v=v: lw_kernel_cuda.flash_attention(q, k, v)
        our_ms, our_std = bu.measure_gpu_time(our_fn, warmup=5, iters=50)

        # Reference: PyTorch's scaled_dot_product_attention
        ref_fn = lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=False)
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn, warmup=5, iters=50)

        flops = bu.attention_flops(B, H, S, D)
        tflops = bu.compute_tflops(flops, our_ms)
        ref_tflops = bu.compute_tflops(flops, ref_ms)

        read_bytes = 3 * B * H * S * D * 4
        write_bytes = B * H * S * D * 4
        bytes_ = read_bytes + write_bytes

        results.append(bu.BenchmarkResult(
            op_name="flash_attn",
            shape_desc=f"B{B}H{H}S{S}D{D}",
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
    bu.run_benchmarks(bench_flash_attention)