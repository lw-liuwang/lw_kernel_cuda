import torch
import sys
sys.path.insert(0, "/workspace/lw-kernel-cuda")
import lw_kernel_cuda

x = torch.tensor([0.5, 0.5], device="cuda", dtype=torch.float32)
print("Calling histogram...")
bins = torch.ops.lw_kernel_cuda.histogram(x, 5, 0.0, 1.0)
print("bins =", bins)
