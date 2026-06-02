import torch
import pytest


def test_softmax_small():
    device = "cuda"
    x = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=torch.float32)
    expected = torch.softmax(x, dim=-1)
    out = torch.ops.lw_kernel_cuda.softmax(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_softmax_batch():
    device = "cuda"
    x = torch.randn(4, 8, device=device, dtype=torch.float32)
    expected = torch.softmax(x, dim=-1)
    out = torch.ops.lw_kernel_cuda.softmax(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_softmax_large():
    device = "cuda"
    x = torch.randn(32, 4096, device=device, dtype=torch.float32)
    expected = torch.softmax(x, dim=-1)
    out = torch.ops.lw_kernel_cuda.softmax(x)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_softmax_sums_to_one():
    device = "cuda"
    x = torch.randn(8, 256, device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.softmax(x)
    sums = out.sum(dim=-1)
    expected = torch.ones(8, device=device, dtype=torch.float32)
    torch.testing.assert_close(sums, expected, rtol=1e-5, atol=1e-5)


def test_softmax_zero():
    device = "cuda"
    x = torch.zeros(2, 5, device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.softmax(x)
    expected = torch.full_like(x, 1.0 / 5.0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)