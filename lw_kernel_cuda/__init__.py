# lw-kernel-cuda Python package
#
# Usage:
#   import lw_kernel_cuda
#   out = lw_kernel_cuda.vec_add(a, b)
#   out = lw_kernel_cuda.softmax(x)
#   ...

from . import _C

# Import all ops so they are accessible as lw_kernel_cuda.<op_name>
from .ops.vec_add import vec_add
from .ops.reduce import reduce
from .ops.softmax import softmax
from .ops.rmsnorm import rmsnorm
from .ops.transpose import transpose
from .ops.matmul import matmul
from .ops.conv2d import conv2d
from .ops.histogram import histogram
from .ops.prefix_sum import prefix_sum
from .ops.flash_attention import flash_attention

__all__ = [
    "vec_add",
    "reduce",
    "softmax",
    "rmsnorm",
    "transpose",
    "matmul",
    "conv2d",
    "histogram",
    "prefix_sum",
    "flash_attention",
]