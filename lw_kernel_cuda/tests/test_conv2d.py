import torch
import pytest


@pytest.mark.parametrize("N, C, H, W, OC, K, S, P", [
    (1, 1, 4, 4, 2, 3, 1, 0),
    (2, 2, 8, 8, 4, 3, 1, 1),
    (1, 3, 8, 8, 6, 3, 1, 0),
])
def test_conv2d(N, C, H, W, OC, K, S, P):
    device = "cuda"
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float32)
    w = torch.randn(OC, C, K, K, device=device, dtype=torch.float32)
    bias = torch.randn(OC, device=device, dtype=torch.float32)

    expected = torch.nn.functional.conv2d(x, w, bias, stride=S, padding=P)
    out = torch.ops.lw_kernel_cuda.conv2d(x, w, bias, S, P)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_conv2d_no_bias():
    device = "cuda"
    x = torch.randn(1, 1, 5, 5, device=device, dtype=torch.float32)
    w = torch.randn(2, 1, 3, 3, device=device, dtype=torch.float32)
    bias = torch.tensor([], device=device, dtype=torch.float32)

    expected = torch.nn.functional.conv2d(x, w, stride=1, padding=0)
    out = torch.ops.lw_kernel_cuda.conv2d(x, w, bias, 1, 0)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)