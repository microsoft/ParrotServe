# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Setup scripts."""

import os
import subprocess
from typing import Set
from packaging.version import parse, Version
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # , CUDA_HOME

# https://github.com/pytorch/pytorch/issues/22844
# HACK(chaofan): Sometimes this method fails to detect correct CUDA version.
# We use environment variable CUDA_HOME instead.

CUDA_HOME = os.getenv("CUDA_HOME")

ROOT_DIR = os.path.dirname(__file__)

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
NVCC_FLAGS = ["-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        f"Cannot find CUDA_HOME. CUDA must be available to build the package."
    )


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


# Collect the compute capabilities of all available GPUs.
device_count = torch.cuda.device_count()
compute_capabilities: Set[int] = set()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 7:
        raise RuntimeError(
            "GPUs with compute capability less than 7.0 are not supported."
        )
    compute_capabilities.add(major * 10 + minor)

# Validate the NVCC CUDA version.
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if 86 in compute_capabilities and nvcc_cuda_version < Version("11.1"):
    raise RuntimeError(
        "CUDA 11.1 or higher is required for GPUs with compute capability 8.6."
    )
if 89 in compute_capabilities and nvcc_cuda_version < Version("11.8"):
    # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
    # However, GPUs with compute capability 8.9 can also run the code generated by
    # the previous versions of CUDA 11 and targeting compute capability 8.0.
    # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
    # instead of 8.9.
    compute_capabilities.remove(89)
    compute_capabilities.add(80)
if 90 in compute_capabilities and nvcc_cuda_version < Version("11.8"):
    raise RuntimeError(
        "CUDA 11.8 or higher is required for GPUs with compute capability 9.0."
    )

# If no GPU is available, add all supported compute capabilities.
if not compute_capabilities:
    compute_capabilities = {70, 75, 80}
    if nvcc_cuda_version >= Version("11.1"):
        compute_capabilities.add(86)
    if nvcc_cuda_version >= Version("11.8"):
        compute_capabilities.add(89)
        compute_capabilities.add(90)

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    NVCC_FLAGS += ["-gencode", f"arch=compute_{capability},code=sm_{capability}"]

# Use NVCC threads to parallelize the build.
if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 8)
    NVCC_FLAGS += ["--threads", str(num_threads)]

ext_modules = []

# Attention kernels.
attention_extension = CUDAExtension(
    name="parrot.attention_ops",
    sources=[
        "csrc/attention.cpp",
        "csrc/attention/attention_kernels.cu",
        "csrc/attention/attention_prev_kernels.cu",
        "csrc/attention/attention_post_kernels.cu",
    ],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(attention_extension)

setup(
    name="parrot",
    version="0.1",
    author="Chaofan Lin",
    package_dir={"": "."},
    packages=find_packages(exclude=("csrc")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
