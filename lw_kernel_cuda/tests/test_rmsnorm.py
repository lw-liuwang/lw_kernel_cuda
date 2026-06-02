import torch
import pytest


def _rmsnorm_ref(x, weight, eps=1e-6):
    return torch.nn.functional.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)


def test_rmsnorm_small():
    device = "cuda"
    x = torch.randn(2, 8, device=device, dtype=torch.float32)
    w = torch.ones(8, device=device, dtype=torch.float32)
    expected = _rmsnorm_ref(x, w)
    out = torch.ops.lw_kernel_cuda.rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_rmsnorm_batch():
    device = "cuda"
    x = torch.randn(16, 128, device=device, dtype=torch.float32)
    w = torch.randn(128, device=device, dtype=torch.float32)
    expected = _rmsnorm_ref(x, w)
    out = torch.ops.lw_kernel_cuda.rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_rmsnorm_large():
    device = "cuda"
    x = torch.randn(64, 4096, device=device, dtype=torch.float32)
    w = torch.randn(4096, device=device, dtype=torch.float32)
    expected = _rmsnorm_ref(x, w)
    out = torch.ops.lw_kernel_cuda.rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_rmsnorm_eps():
    device = "cuda"
    x = torch.zeros(2, 16, device=device, dtype=torch.float32)
    w = torch.ones(16, device=device, dtype=torch.float32)
    # With eps=1.0, output should be near zero (since denom is large)
    out = torch.ops.lw_kernel_cuda.rmsnorm(x, w, 1.0)
    expected = _rmsnorm_ref(x, w, 1.0)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)