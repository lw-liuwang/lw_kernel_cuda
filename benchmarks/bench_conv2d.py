"""Benchmark Conv2D operator vs F.conv2d. Reports TFLOPS."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import lw_kernel_cuda

from . import bench_utils as bu


def bench_conv2d() -> list[bu.BenchmarkResult]:
    torch.cuda.synchronize()
    results: list[bu.BenchmarkResult] = []

    # ResNet-like configs: (N, IC, OC, H, W, KH, KW, stride, padding)
    configs = [
        # ResNet18/34 first layer
        (1, 3, 64, 224, 224, 7, 7, 2, 3),
        # ResNet18/34 conv1
        (1, 64, 64, 56, 56, 3, 3, 1, 1),
        # ResNet18/34 conv2
        (1, 64, 128, 56, 56, 3, 3, 2, 1),
        # ResNet18/34 conv3
        (1, 128, 256, 28, 28, 3, 3, 2, 1),
        # ResNet18/34 conv4
        (1, 256, 512, 14, 14, 3, 3, 2, 1),
        # Batch inference
        (4, 3, 64, 224, 224, 7, 7, 2, 3),
        (4, 64, 64, 56, 56, 3, 3, 1, 1),
        # 1x1 conv (pointwise)
        (1, 256, 64, 56, 56, 1, 1, 1, 0),
        (1, 512, 128, 28, 28, 1, 1, 1, 0),
        # Depthwise-like (IC == OC == groups)
        (1, 64, 64, 56, 56, 3, 3, 1, 1),
    ]

    for (N, IC, OC, H, W, KH, KW, stride, padding) in configs:
        x = torch.randn(N, IC, H, W, device="cuda", dtype=torch.float32)
        weight = torch.randn(OC, IC, KH, KW, device="cuda", dtype=torch.float32)

        # Compute output spatial dimensions
        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1

        # Our implementation
        our_fn = lambda x=x, w=weight: lw_kernel_cuda.conv2d(x, w, None, stride, padding)
        our_ms, our_std = bu.measure_gpu_time(our_fn)

        # Reference: F.conv2d
        ref_fn = lambda x=x, w=weight: F.conv2d(x, w, padding=padding, stride=stride)
        ref_ms, ref_std = bu.measure_gpu_time(ref_fn)

        flops = bu.conv2d_flops(N, OC, IC, OH, OW, KH, KW)
        tflops = bu.compute_tflops(flops, our_ms)
        ref_tflops = bu.compute_tflops(flops, ref_ms)

        read_bytes = N * IC * H * W * 4 + OC * IC * KH * KW * 4
        write_bytes = N * OC * OH * OW * 4
        bytes_ = read_bytes + write_bytes

        shape_label = f"{N}x{IC}x{H}x{W}_{OC}x{IC}x{KH}x{KW}"
        results.append(bu.BenchmarkResult(
            op_name="conv2d",
            shape_desc=shape_label,
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
    bu.run_benchmarks(bench_conv2d)