import torch, sys
sys.path.insert(0, "/workspace/lw-kernel-cuda")
import lw_kernel_cuda

# Test with empty tensor
x = torch.tensor([], device="cuda", dtype=torch.float32)
try:
    bins = torch.ops.lw_kernel_cuda.histogram(x, 5, 0.0, 1.0)
    print("Empty test PASSED, bins =", bins)
except Exception as e:
    print("Empty test FAILED:", e)
