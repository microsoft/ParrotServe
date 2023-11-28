# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import torch

from vllm import pos_encoding_ops


def vllm_rotary_emb(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
):
    """Neo-X style rotary embedding."""

    pos_encoding_ops.rotary_embedding_neox(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
    )
