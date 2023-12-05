# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum, auto


class MemLayout(Enum):
    """Memory layout for KV cache."""

    NORMAL: int = auto()  # [head_num, head_size,]
    BLOCK: int = auto()  # [head_num, head_size, block_size]
    VLLM: int = (
        auto()
    )  # k: [head_num, head_size // x, block_size, x], v: [head_num, head_size, block_size]


ATTN_FUNC_LAYOUT_MAP = {
    "xformers_with_buffer": MemLayout.NORMAL,
    "xformers_fill_vllm_paged_attention_generate": MemLayout.VLLM,
    "xformers_fill_shared_prompts_generate": MemLayout.VLLM,
}
