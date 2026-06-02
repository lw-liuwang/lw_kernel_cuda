import torch


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax along the last dimension (row-wise).

    Args:
        x: Input tensor (float32, CUDA) of shape (rows, cols)

    Returns:
        out: Softmax-normalized tensor (float32, CUDA)
    """
    return torch.ops.lw_kernel_cuda.softmax(x)