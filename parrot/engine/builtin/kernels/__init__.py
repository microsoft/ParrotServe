# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Kernels for Parrot built-in engines."""

from .tokens_moving import (
    discontinuous_move_tokens,
    move_tokens_from_blocked_k_cache,
    move_tokens_from_blocked_v_cache,
)
from .rotary_embedding import rotary_embedding
from .rms_norm import rmsnorm_forward

from .vllm import *
from .shared_flash_decoding import flash_paged_attention, paged_flash_attention
