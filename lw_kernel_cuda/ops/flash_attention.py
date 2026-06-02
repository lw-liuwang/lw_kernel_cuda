import torch


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """FlashAttention: tiled attention with online softmax.

    Args:
        q: Query tensor (float32, CUDA) of shape (B, H, seqlen, dim)
        k: Key tensor (float32, CUDA) of shape (B, H, seqlen, dim)
        v: Value tensor (float32, CUDA) of shape (B, H, seqlen, dim)

    Returns:
        out: Attention output (float32, CUDA) of shape (B, H, seqlen, dim)
    """
    return torch.ops.lw_kernel_cuda.flash_attention(q, k, v)