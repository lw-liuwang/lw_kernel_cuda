import os
import glob

from setuptools import find_packages, setup

# Monkey-patch CUDA version check before any torch internals
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None

import torch
from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "lw_kernel_cuda"


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "lw_kernel_cuda", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    include_paths = [os.path.join(this_dir, "include")]

    ext_modules = [
        CUDAExtension(
            f"{library_name}._C",
            sources,
            include_dirs=include_paths,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version="0.1.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    description="lw-kernel-cuda: CUDA operator library built from scratch",
    cmdclass={"build_ext": BuildExtension},
)