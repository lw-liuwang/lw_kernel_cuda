import torch
import pytest


def test_matmul_small():
    device = "cuda"
    a = torch.randn(4, 8, device=device, dtype=torch.float32)
    b = torch.randn(8, 4, device=device, dtype=torch.float32)
    expected = a @ b
    out = torch.ops.lw_kernel_cuda.matmul(a, b)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_matmul_rect():
    device = "cuda"
    a = torch.randn(16, 32, device=device, dtype=torch.float32)
    b = torch.randn(32, 48, device=device, dtype=torch.float32)
    expected = a @ b
    out = torch.ops.lw_kernel_cuda.matmul(a, b)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_matmul_large():
    device = "cuda"
    a = torch.randn(256, 256, device=device, dtype=torch.float32)
    b = torch.randn(256, 256, device=device, dtype=torch.float32)
    expected = a @ b
    out = torch.ops.lw_kernel_cuda.matmul(a, b)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_matmul_identity():
    device = "cuda"
    a = torch.randn(16, 16, device=device, dtype=torch.float32)
    eye = torch.eye(16, device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.matmul(a, eye)
    torch.testing.assert_close(out, a, rtol=1e-4, atol=1e-4)


def test_matmul_zero():
    device = "cuda"
    a = torch.zeros(8, 12, device=device, dtype=torch.float32)
    b = torch.randn(12, 6, device=device, dtype=torch.float32)
    expected = torch.zeros(8, 6, device=device, dtype=torch.float32)
    out = torch.ops.lw_kernel_cuda.matmul(a, b)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)