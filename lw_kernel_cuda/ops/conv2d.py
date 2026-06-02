import torch


def conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """2D Convolution (im2col + GEMM).

    Args:
        x: Input tensor (float32, CUDA) of shape (N, C, H, W)
        weight: Weight tensor (float32, CUDA) of shape (OC, IC, KH, KW)
        bias: Optional bias tensor (float32, CUDA) of shape (OC,)
        stride: Convolution stride
        padding: Convolution padding

    Returns:
        out: Output tensor (float32, CUDA) of shape (N, OC, OH, OW)
    """
    bias_tensor = bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype)
    return torch.ops.lw_kernel_cuda.conv2d(x, weight, bias_tensor, stride, padding)