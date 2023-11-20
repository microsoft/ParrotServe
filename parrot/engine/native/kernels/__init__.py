# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Kernels for Parrot native engines."""

from .tokens_moving import (
    discontinuous_move_tokens,
    move_tokens_from_blocked_k_cache,
    move_tokens_from_blocked_v_cache,
)
from .rotary_embedding import rotary_embedding
from .paged_attention import vllm_paged_attention, vllm_reshape_and_cache
from .rms_norm import rmsnorm_forward
