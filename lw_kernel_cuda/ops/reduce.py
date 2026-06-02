import torch


def reduce(x: torch.Tensor) -> torch.Tensor:
    """Reduce (sum) all elements of a tensor.

    Args:
        x: Input tensor (float32, CUDA)

    Returns:
        out: Scalar tensor of sum(x)
    """
    return torch.ops.lw_kernel_cuda.reduce(x)