import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root Mean Square Layer Normalization.

    Args:
        x: Input tensor (float32, CUDA) of shape (rows, cols)
        weight: Weight tensor (float32, CUDA) of shape (cols,)
        eps: Small constant for numerical stability

    Returns:
        out: Normalized tensor (float32, CUDA)
    """
    return torch.ops.lw_kernel_cuda.rmsnorm(x, weight, eps)