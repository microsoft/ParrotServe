from typing import List, Optional

import torch
from ..utils import get_logger

logger = get_logger("Mem")


class KVContext:
    """Low-level implementation of Context."""

    def __init__(self, context_id: int, parent_context: Optional["KVContext"]):
        self.context_id = context_id
        self.parent_context = parent_context
        self.tokens_kv_block_id: List[int] = []
        self.tokens_id: List[int] = []

    def get_context_len(self) -> int:
        """Return the length of the context."""
        parent_len = self.parent_context.context_len if self.parent_context else 0
        return parent_len + len(self.tokens_kv_block_id)

    def get_context_blocks(self) -> List[int]:
        """Return the context blocks."""
        parent_blocks = (
            self.parent_context.get_context_blocks() if self.parent_context else []
        )
        return parent_blocks + self.tokens_kv_block_id


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
        # logger.info(
        #     f"Allocated {blocks_num} blocks. Total size: {blocks_num * num_heads * head_size / 1024 / 1024 / 1024 :.2f} GiB"
        # )

        self.storage = torch.empty(
            [blocks_num, num_heads, head_size],
            dtype=dtype,
            device=device,
        )
