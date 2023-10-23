from typing import List, Optional
import torch

from parrot.utils import RecyclePool

from ..low_level_context import LowLevelContext


class BlockContext(LowLevelContext):
    """BlockContext: Use the idea of PagedAttention to manage the memory."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["BlockContext"],
        kv_cache_manager: RecyclePool,
    ):
        super().__init__(context_id, parent_context)

        # KV blocks address
        self.token_kv_block_ids: List[int] = []
        self.token_ids: List[int] = []

        # KV cache manager i.e. a pool allocator.
        self.kv_cache_manager = kv_cache_manager

        # If the context is extended by the `fill` primitive, it should has a
        # `last_hidden_state` for the `generation` primitive.
        self.last_hidden_state: Optional[torch.Tensor] = None

    def destruction(self):
        super().destruction()

        for block_id in self.token_kv_block_ids:
            self.kv_cache_manager.free(block_id)

    def allocate(self, length: int):
        for _ in range(length):
            self.token_kv_block_ids.append(self.kv_cache_manager.allocate())

    def get_context_len(self) -> int:
        parent_len = self.parent_context.get_context_len() if self.parent_context else 0
        return parent_len + self.get_this_context_len()

    def get_this_context_len(self) -> int:
        return len(self.token_kv_block_ids)

    def get_context_blocks(self) -> List[int]:
        """Return the context blocks."""

        parent_blocks = (
            self.parent_context.get_context_blocks() if self.parent_context else []
        )
        return parent_blocks + self.token_kv_block_ids

    def get_last_token_id(self) -> int:
        """Return the last token id."""

        return self.token_ids[-1]

    def get_last_hidden_state(self) -> torch.Tensor:
        """Return the last hidden state."""

        if len(self.token_ids) == 0:
            assert (
                self.parent_context is not None
            ), "The parent context should not be None if this is an empty context."
            return self.parent_context.get_last_hidden_state()

        return self.last_hidden_state
