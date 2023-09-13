from typing import List
from dataclasses import dataclass

import torch


@dataclass
class KVContext:
    """Low-level implementation of Context."""

    context_id: int
    tokens_block_id: List[int]
    last_token_id: int
    parent_context: "KVContext"

    def get_context_len(self) -> int:
        """Return the length of the context."""

        return self.parent_context.context_len + len(self.tokens_block_id)

    def get_context_blocks(self) -> List[int]:
        """Return the context blocks."""

        return self.parent_context.get_context_blocks() + self.tokens_block_id


class KVCacheStorage:
    """KV cache storage in one Attention layer."""

    def __init__(
        self,
        blocks_num: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.storage = torch.empty(
            [blocks_num, num_heads, head_size],
            dtype=dtype,
            device=device,
        )
