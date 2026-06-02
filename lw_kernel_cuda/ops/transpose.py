import torch


def transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D matrix.

    Args:
        x: Input matrix (float32, CUDA) of shape (rows, cols)

    Returns:
        out: Transposed matrix (float32, CUDA) of shape (cols, rows)
    """
    return torch.ops.lw_kernel_cuda.transpose(x)