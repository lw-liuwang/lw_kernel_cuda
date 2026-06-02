"""
Core benchmark infrastructure for lw-kernel-cuda.

Provides:
- GPUEventTimer: GPU-side precise timing via torch.cuda.Event
- BenchmarkResult: dataclass for storing results
- compute_bandwidth_gbps / compute_tflops: metric helpers
- format_results: pretty-print results table
- export_csv: write results to CSV
"""

from __future__ import annotations

import csv
import dataclasses
import math
import sys
import time
from typing import Callable, Optional

import torch

# ---------------------------------------------------------------------------
# Hardware parameters (NVIDIA A10, SM86)
# ---------------------------------------------------------------------------
DEVICE_NAME: str = "NVIDIA A10"
SM_COUNT: int = 72
MEMORY_BW_GBPS: float = 600.0  # theoretical memory bandwidth (GB/s)
FP32_TFLOPS: float = 31.2  # theoretical peak FP32 TFLOPS

WARMUP_ITERS: int = 10
MEASURED_ITERS: int = 100


# ---------------------------------------------------------------------------
# GPU Event Timer
# ---------------------------------------------------------------------------

class GPUEventTimer:
    """Precise GPU-side timer using torch.cuda.Event.

    Usage:
        timer = GPUEventTimer()
        timer.start()
        kernel_fn()
        timer.end()
        ms = timer.elapsed_ms()
    """

    def __init__(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.start_event.record()

    def end(self) -> None:
        self.end_event.record()

    def elapsed_ms(self) -> float:
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def measure_gpu_time(fn: Callable, *args, warmup: int = WARMUP_ITERS,
                     iters: int = MEASURED_ITERS, **kwargs) -> tuple[float, float]:
    """Measure GPU kernel execution time.

    Args:
        fn: Callable (kernel or lambda).
        warmup: Number of warm-up iterations (not measured).
        iters: Number of measured iterations.

    Returns:
        (mean_ms, std_ms)
    """
    # Warm-up
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Measured runs
    times_ms = []
    for _ in range(iters):
        timer = GPUEventTimer()
        timer.start()
        fn(*args, **kwargs)
        timer.end()
        times_ms.append(timer.elapsed_ms())

    mean_ms = float(torch.tensor(times_ms).mean().item())
    std_ms = float(torch.tensor(times_ms).std().item())
    return mean_ms, std_ms


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BenchmarkResult:
    op_name: str
    shape_desc: str
    our_ms: float
    our_std: float
    ref_ms: float
    ref_std: float
    metric: float  # GB/s or TFLOPS for our impl
    ref_metric: float  # GB/s or TFLOPS for reference impl
    metric_name: str  # "GB/s" or "TFLOPS"
    bytes_processed: int = 0
    flops: int = 0


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_bandwidth_gbps(bytes_: int, time_ms: float) -> float:
    """Compute effective bandwidth in GB/s."""
    if time_ms <= 0:
        return 0.0
    return (bytes_ / 1e9) / (time_ms / 1e3)


def compute_tflops(flops: int, time_ms: float) -> float:
    """Compute effective TFLOPS."""
    if time_ms <= 0:
        return 0.0
    return (flops / 1e12) / (time_ms / 1e3)


def format_metric(metric_name: str, val: float) -> str:
    """Format a metric value with appropriate precision."""
    if metric_name == "TFLOPS":
        return f"{val:.2f}"
    else:
        return f"{val:.0f}"


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a pretty-printed table.

    The table adapts columns based on the metric type (GB/s vs TFLOPS).
    """
    if not results:
        return ""

    # Determine metric type from first result
    use_tflops = results[0].metric_name == "TFLOPS"

    # Column definitions
    if use_tflops:
        headers = ["Op", "Shape", "Ours(ms)", "Ref(ms)", "Speedup",
                    "Ours(TFLOPS)", "Ref(TFLOPS)"]
        col_widths = [16, 16, 10, 10, 9, 14, 14]
        fmt_row = (
            "{:<16} {:<16} {:>8.3f} {:>8.3f}  {:>7.2f}x  {:>12.2f}  {:>12.2f}"
        )
    else:
        headers = ["Op", "Shape", "Ours(us)", "Ref(us)", "Speedup",
                    "Ours(GB/s)", "Ref(GB/s)"]
        col_widths = [16, 16, 10, 10, 9, 14, 14]
        fmt_row = (
            "{:<16} {:<16} {:>8.1f} {:>8.1f}  {:>7.2f}x  {:>12.0f}  {:>12.0f}"
        )

    separator = "-" * (sum(col_widths) + 10)

    lines = [separator]
    header_line = "  ".join(h.center(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append(separator)

    for r in results:
        speedup = r.ref_ms / r.our_ms if r.our_ms > 0 else 0.0
        if use_tflops:
            line = fmt_row.format(
                r.op_name, r.shape_desc,
                r.our_ms, r.ref_ms, speedup,
                r.metric, r.ref_metric,
            )
        else:
            # Convert ms -> us for display
            line = fmt_row.format(
                r.op_name, r.shape_desc,
                r.our_ms * 1000, r.ref_ms * 1000, speedup,
                r.metric, r.ref_metric,
            )
        lines.append(line)

    lines.append(separator)
    lines.append(f"  Device: {DEVICE_NAME}")
    lines.append(f"  Theoretical Peak: {MEMORY_BW_GBPS:.0f} GB/s, {FP32_TFLOPS:.1f} TFLOPS")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(results: list[BenchmarkResult], filepath: str) -> None:
    """Export benchmark results to CSV."""
    fieldnames = [
        "op_name", "shape_desc", "our_ms", "our_std_ms",
        "ref_ms", "ref_std_ms", "metric", "ref_metric",
        "metric_name", "bytes_processed", "flops",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "op_name": r.op_name,
                "shape_desc": r.shape_desc,
                "our_ms": f"{r.our_ms:.6f}",
                "our_std_ms": f"{r.our_std:.6f}",
                "ref_ms": f"{r.ref_ms:.6f}",
                "ref_std_ms": f"{r.ref_std:.6f}",
                "metric": f"{r.metric:.4f}",
                "ref_metric": f"{r.ref_metric:.4f}",
                "metric_name": r.metric_name,
                "bytes_processed": str(r.bytes_processed),
                "flops": str(r.flops),
            })
    print(f"  Results exported to {filepath}")


# ---------------------------------------------------------------------------
# Running all benchmarks
# ---------------------------------------------------------------------------

def run_benchmarks(bench_fn: Callable) -> list[BenchmarkResult]:
    """Wrapper to run a benchmark function and measure elapsed time."""
    print(f"\n{'=' * 70}")
    print(f"Running: {bench_fn.__name__}")
    print(f"{'=' * 70}")
    t0 = time.time()
    results = bench_fn()
    elapsed = time.time() - t0
    print(format_results(results))
    print(f"  Elapsed: {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# FLOPS / bytes helpers for common operators
# ---------------------------------------------------------------------------

def matmul_flops(M: int, N: int, K: int) -> int:
    """2 * M * N * K (multiply + add)."""
    return 2 * M * N * K


def matmul_bytes(M: int, N: int, K: int) -> int:
    """Read A (M*K), B (K*N), write C (M*N) in float32."""
    return 4 * (M * K + K * N + M * N)


def softmax_flops(rows: int, cols: int) -> int:
    """exp + add + div per element, plus max+sum per row."""
    # Per element: max sub (1), exp (1), sum (1), div (1) = 4 ops
    # Per row: max reduce (cols), sum reduce (cols) = 2*cols
    return rows * (4 * cols + 2 * cols)  # simplified: 6 * rows * cols


def softmax_bytes(rows: int, cols: int) -> int:
    """Read x, write out: 2 * rows * cols * 4."""
    return 2 * rows * cols * 4


def rmsnorm_flops(rows: int, cols: int) -> int:
    """mean(sum(x*x)), sqrt, x/mean * weight = ~3*rows*cols + rows*cols."""
    return 4 * rows * cols


def rmsnorm_bytes(rows: int, cols: int) -> int:
    """Read x, weight, write out: (rows*cols + cols + rows*cols) * 4."""
    return 4 * (rows * cols + cols + rows * cols)


def attention_flops(B: int, H: int, S: int, D: int) -> int:
    """QK^T (B*H*S*S*2*D) + softmax (~6*S) + PV (B*H*S*S*2*D)."""
    # Actually: Q@K.T = 2*B*H*S*S*D, P@V = 2*B*H*S*S*D, softmax overhead
    return 4 * B * H * S * S * D


def conv2d_flops(N: int, OC: int, IC: int, OH: int, OW: int, KH: int, KW: int) -> int:
    """2 * N * OC * OH * OW * IC * KH * KW."""
    return 2 * N * OC * OH * OW * IC * KH * KW