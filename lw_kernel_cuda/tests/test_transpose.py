import torch
import pytest


def test_transpose_small():
    device = "cuda"
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device, dtype=torch.float32)
    expected = x.t().contiguous()
    out = torch.ops.lw_kernel_cuda.transpose(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_transpose_square():
    device = "cuda"
    x = torch.randn(32, 32, device=device, dtype=torch.float32)
    expected = x.t().contiguous()
    out = torch.ops.lw_kernel_cuda.transpose(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_transpose_rect():
    device = "cuda"
    x = torch.randn(16, 64, device=device, dtype=torch.float32)
    expected = x.t().contiguous()
    out = torch.ops.lw_kernel_cuda.transpose(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_transpose_large():
    device = "cuda"
    x = torch.randn(512, 512, device=device, dtype=torch.float32)
    expected = x.t().contiguous()
    out = torch.ops.lw_kernel_cuda.transpose(x)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(1, 1), (1, 100), (100, 1), (127, 63)])
def test_transpose_various(shape):
    device = "cuda"
    x = torch.randn(*shape, device=device, dtype=torch.float32)
    expected = x.t().contiguous()
    out = torch.ops.lw_kernel_cuda.transpose(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)