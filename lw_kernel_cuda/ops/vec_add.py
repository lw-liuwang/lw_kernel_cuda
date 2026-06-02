import torch


def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of two tensors on GPU.

    Args:
        a: Input tensor (float32, CUDA)
        b: Input tensor (float32, CUDA), same shape as a

    Returns:
        out: a + b (float32, CUDA)
    """
    return torch.ops.lw_kernel_cuda.vec_add(a, b)