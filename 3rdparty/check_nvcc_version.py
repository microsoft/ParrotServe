import subprocess
from packaging.version import parse, Version

# from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
# print(CUDA_HOME)

import os

CUDA_HOME = os.getenv("CUDA_HOME")
print(CUDA_HOME)


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


print(get_nvcc_cuda_version(CUDA_HOME))
