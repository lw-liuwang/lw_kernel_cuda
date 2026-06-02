import torch


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication: C = A @ B.

    Args:
        a: Matrix A (float32, CUDA) of shape (M, K)
        b: Matrix B (float32, CUDA) of shape (K, N)

    Returns:
        out: Matrix C (float32, CUDA) of shape (M, N)
    """
    return torch.ops.lw_kernel_cuda.matmul(a, b)