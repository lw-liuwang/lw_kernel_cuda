import torch
import pytest


def test_vec_add():
    device = "cuda"
    for n in [1, 10, 100, 1000, 10000, 100000]:
        a = torch.randn(n, device=device, dtype=torch.float32)
        b = torch.randn(n, device=device, dtype=torch.float32)
        expected = a + b
        out = torch.ops.lw_kernel_cuda.vec_add(a, b)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_vec_add_large():
    device = "cuda"
    n = 1000000
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)
    expected = a + b
    out = torch.ops.lw_kernel_cuda.vec_add(a, b)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_vec_add_empty():
    device = "cuda"
    a = torch.zeros(0, device=device, dtype=torch.float32)
    b = torch.zeros(0, device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.vec_add(a, b)
    assert out.numel() == 0


def test_vec_add_constant():
    device = "cuda"
    a = torch.ones(100, device=device, dtype=torch.float32)
    b = torch.ones(100, device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.vec_add(a, b)
    expected = torch.full_like(a, 2.0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)