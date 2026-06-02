import torch
import pytest


def test_flash_attention_small():
    device = "cuda"
    B, H, seqlen, dim = 1, 1, 4, 128
    q = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)
    k = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)
    v = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)

    out = torch.ops.lw_kernel_cuda.flash_attention(q, k, v)

    # Reference: PyTorch SDP attention
    scale = dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = attn @ v

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_flash_attention_multihead():
    device = "cuda"
    B, H, seqlen, dim = 2, 4, 8, 128
    q = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)
    k = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)
    v = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)

    out = torch.ops.lw_kernel_cuda.flash_attention(q, k, v)

    scale = dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    expected = attn @ v

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_flash_attention_causal_property():
    """Just verify the kernel runs and produces finite outputs."""
    device = "cuda"
    B, H, seqlen, dim = 1, 1, 8, 128
    q = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)
    k = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)
    v = torch.randn(B, H, seqlen, dim, device=device, dtype=torch.float32)

    out = torch.ops.lw_kernel_cuda.flash_attention(q, k, v)
    assert torch.isfinite(out).all()