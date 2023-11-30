# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import inspect
import sys
import os
import psutil


def redirect_stdout_stderr_to_file(log_file_dir_path: str, file_name: str):
    """Redirect stdout and stderr to a file."""

    path = os.path.join(log_file_dir_path, file_name)

    counter = 1
    orig_filename = file_name
    while os.path.exists(path):
        file_name = orig_filename + str(counter)
        path = os.path.join(log_file_dir_path, file_name)
        counter += 1

    fp = open(path, "w+")
    sys.stdout = fp
    sys.stderr = fp


def set_random_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def change_signature(func, new_parameters, new_return_annotation):
    """Change a function's signature.

    Reference: https://deepinout.com/python/python-qa/369_python_set_function_signature_in_python.html
    """

    signature = inspect.signature(func)
    new_signature = signature.replace(
        parameters=new_parameters,
        return_annotation=new_return_annotation,
    )
    func.__signature__ = new_signature


def get_cpu_memory_usage() -> float:
    """Get the current process's CPU memory usage in MiB."""

    # process = psutil.Process(os.getpid())
    # return process.memory_info().rss / 1024 / 1024
    return psutil.virtual_memory().used / 1024 / 1024
