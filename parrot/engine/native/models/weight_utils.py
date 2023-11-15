# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/weight_utils.py
# Copyright 2023 The vLLM team.

"""Utilities for downloading and initializing model weights."""
import filelock
import glob
import os
from typing import Iterator, Tuple

from huggingface_hub import snapshot_download
import torch


def hf_weights_loader(model_name: str) -> Iterator[Tuple[str, torch.Tensor]]:
    # Prepare file lock directory to prevent multiple processes from
    # downloading the same model weights at the same time.
    lock_dir = "/tmp"
    lock_file_name = model_name.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))

    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name)
    if not is_local:
        with lock:
            hf_folder = snapshot_download(
                model_name,
                allow_patterns="*.bin",
            )
    else:
        hf_folder = model_name

    # Glob bin files.
    hf_bin_files = [
        x
        for x in glob.glob(os.path.join(hf_folder, "*.bin"))
        if not x.endswith("training_args.bin")
    ]

    for bin_file in hf_bin_files:
        state = torch.load(bin_file, map_location="cpu", weights_only=True)
        for name, param in state.items():
            yield name, param
        del state
        torch.cuda.empty_cache()
