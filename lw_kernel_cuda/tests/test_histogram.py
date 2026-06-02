import torch
import pytest


def test_histogram_uniform():
    device = "cuda"
    # Uniform values should produce roughly equal bin counts.
    # Use rand to avoid endpoint exclusion issue (linspace includes max_val
    # which is excluded by the [min_val, max_val) range).
    n = 10000
    x = torch.rand(n, device=device, dtype=torch.float32)  # uniform in [0, 1)
    bins = torch.ops.lw_kernel_cuda.histogram(x, 10, 0.0, 1.0)
    assert bins.sum().item() == n
    # Each bin should have roughly n/10 elements
    expected_per_bin = n / 10
    for i in range(10):
        assert abs(bins[i].item() - expected_per_bin) < expected_per_bin * 0.5


def test_histogram_single_value():
    device = "cuda"
    x = torch.ones(100, device=device, dtype=torch.float32) * 0.5
    bins = torch.ops.lw_kernel_cuda.histogram(x, 10, 0.0, 1.0)
    assert bins.sum().item() == 100
    # Value 0.5 should fall into bin 5 (out of 10)
    assert bins[5].item() == 100


def test_histogram_out_of_range():
    device = "cuda"
    x = torch.tensor([-10.0, 10.0], device=device, dtype=torch.float32)
    bins = torch.ops.lw_kernel_cuda.histogram(x, 5, 0.0, 1.0)
    # Both values are outside [0, 1), so no counts
    assert bins.sum().item() == 0


def test_histogram_large():
    device = "cuda"
    n = 100000
    x = torch.randn(n, device=device, dtype=torch.float32)  # normal distribution
    # Most values should fall within [-3, 3]
    bins = torch.ops.lw_kernel_cuda.histogram(x, 20, -3.0, 3.0)
    assert bins.sum().item() > 0