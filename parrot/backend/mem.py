from typing import List, Optional

import torch
from ..utils import get_logger, RecyclePool

logger = get_logger("Mem")


class KVContext:
    """Low-level implementation of Context."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["KVContext"],
        kv_cache_manager: RecyclePool,
    ):
        self.context_id = context_id
        self.sub_contexts: List["KVContext"] = []

        # Link with parent context
        self.parent_context = parent_context
        parent_context.sub_contexts.append(self) if parent_context else None
        self.tokens_kv_block_id: List[int] = []
        self.token_ids: List[int] = []

        # KV cache manager i.e. a pool allocator.
        self.kv_cache_manager = kv_cache_manager

        # Flag to indicate whether the context is extended by a Fill primitive.
        # If the context is extended by a Fill primitive recently, the last token
        # will be added the next Generation primitive.
        self.last_extended_by_fill = False

    def __del__(self):
        self.parent_context.sub_contexts.remove(self) if self.parent_context else None
        assert len(self.sub_contexts) == 0, "Sub-contexts should be deleted first."
        for block_id in self.tokens_kv_block_id:
            self.kv_cache_manager.free(block_id)

    def allocate(self, length: int):
        for _ in range(length):
            self.tokens_kv_block_id.append(self.kv_cache_manager.allocate())

    def get_context_len(self) -> int:
        """Return the length of the context."""

        parent_len = self.parent_context.get_context_len() if self.parent_context else 0
        return parent_len + len(self.tokens_kv_block_id)

    def get_context_blocks(self) -> List[int]:
        """Return the context blocks."""

        parent_blocks = (
            self.parent_context.get_context_blocks() if self.parent_context else []
        )
        return parent_blocks + self.tokens_kv_block_id

    def get_last_token_id(self) -> int:
        """Return the last token id."""

        return self.token_ids[-1]


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
