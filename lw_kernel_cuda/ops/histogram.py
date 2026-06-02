import torch


def histogram(
    x: torch.Tensor,
    num_bins: int = 10,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Compute histogram of input values.

    Args:
        x: Input tensor (float32, CUDA)
        num_bins: Number of histogram bins
        min_val: Minimum value for binning range
        max_val: Maximum value for binning range

    Returns:
        bins: Integer tensor of bin counts (on CUDA)
    """
    return torch.ops.lw_kernel_cuda.histogram(x, num_bins, min_val, max_val)