import torch
import pytest


def test_prefix_sum_small():
    device = "cuda"
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device, dtype=torch.float32)
    expected = torch.cumsum(x, dim=0)
    out = torch.ops.lw_kernel_cuda.prefix_sum(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_prefix_sum_ones():
    device = "cuda"
    n = 100
    x = torch.ones(n, device=device, dtype=torch.float32)
    expected = torch.arange(1, n + 1, device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.prefix_sum(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_prefix_sum_negative():
    device = "cuda"
    x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], device=device, dtype=torch.float32)
    expected = torch.cumsum(x, dim=0)
    out = torch.ops.lw_kernel_cuda.prefix_sum(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_prefix_sum_single():
    device = "cuda"
    x = torch.tensor([42.0], device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.prefix_sum(x)
    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)