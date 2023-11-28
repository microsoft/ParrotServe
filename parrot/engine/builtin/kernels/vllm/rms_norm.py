# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import torch
from vllm import layernorm_ops


def vllm_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out = torch.empty_like(input)
    layernorm_ops.rms_norm(out, input, weight, eps)
    return out
