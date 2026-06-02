import torch
import pytest


def test_reduce_small():
    device = "cuda"
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device, dtype=torch.float32)
    expected = torch.tensor([15.0], device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.reduce(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_reduce_large():
    device = "cuda"
    n = 1000000
    x = torch.ones(n, device=device, dtype=torch.float32)
    expected = torch.tensor([float(n)], device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.reduce(x)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_reduce_random():
    device = "cuda"
    x = torch.randn(100000, device=device, dtype=torch.float32)
    expected = torch.tensor([x.sum()], device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.reduce(x)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_reduce_negative():
    device = "cuda"
    x = torch.tensor([-1.0, -2.0, -3.0, 10.0], device=device, dtype=torch.float32)
    expected = torch.tensor([4.0], device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.reduce(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)