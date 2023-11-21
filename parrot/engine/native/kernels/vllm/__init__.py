# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

# Kernels from vLLM project.

# For fair evaluation, we need use the same kernels in some irelevant
# components, such as LayerNorm, Rotary Embedding, etc.

from .paged_attention import vllm_paged_attention
from .reshape_and_cache import vllm_reshape_and_cache
from .rms_norm import vllm_rms_norm
from .rotary_embed import vllm_rotary_emb
