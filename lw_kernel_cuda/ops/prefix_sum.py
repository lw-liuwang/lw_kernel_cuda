import torch


def prefix_sum(x: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sum (scan) of a 1D tensor.

    Args:
        x: Input tensor (float32, CUDA) of shape (n,)

    Returns:
        out: Inclusive prefix sum (float32, CUDA) of shape (n,)
    """
    return torch.ops.lw_kernel_cuda.prefix_sum(x)