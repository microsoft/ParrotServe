# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional
import torch

from parrot.utils import RecyclePool

from .low_level_context import LowLevelContext


class BlockContext(LowLevelContext):
    """BlockContext: Use the idea of PagedAttention to manage the memory."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["BlockContext"],
        kv_cache_manager: RecyclePool,
        block_size: int,
    ):
        super().__init__(context_id, parent_context)

        # For blocked context
        self.block_size = block_size
        self.padded = False

        if self.parent_context is not None and not self.parent_context.padded:
            context_len = self.parent_context.get_this_context_len()
            padded_len = (
                (context_len + self.block_size - 1) // self.block_size * self.block_size
            )
            self.parent_context.pad_to(padded_len)

        # KV blocks address
        # length = num_tokens. Each element is a block id.
        # Hence the list is like [0, 0, 0, 1, 1, 1, 2, 2, 2] (block size = 3)
        self.token_kv_block_ids: List[int] = []
        # KV blocks offset. like [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.token_kv_slot_ids: List[int] = []

        # Token ids
        self.token_ids: List[int] = []  # length = num_tokens

        # KV cache manager i.e. a pool allocator.
        self.kv_cache_manager = kv_cache_manager

        # If the context is extended by the `fill` primitive, it should has a
        # `last_hidden_state` for the `generation` primitive.
        self.last_hidden_state: Optional[torch.Tensor] = None

    def _allocate_one(self):
        idx = len(self.token_kv_block_ids)
        if idx % self.block_size == 0:
            cur_block_id = self.kv_cache_manager.allocate()
            self.token_kv_block_ids.append(cur_block_id)
            self.token_kv_slot_ids.append(cur_block_id * self.block_size)
            # print(self.token_kv_block_ids, idx)
        else:
            cur_block_id = self.token_kv_block_ids[-1]
            last_slot_id = self.token_kv_slot_ids[-1]
            self.token_kv_block_ids.append(cur_block_id)
            self.token_kv_slot_ids.append(last_slot_id + 1)

    def pad_to(self, length: int):
        """Pad the context to a certain length."""

        cur_len = self.get_this_context_len()
        assert length >= cur_len, "The length should be larger than the current length."

        for _ in range(length - cur_len):
            self._allocate_one()

        self.padded = True

    # override
    def destruction(self):
        super().destruction()

        for block_id in self.token_kv_block_ids[:: self.block_size]:
            self.kv_cache_manager.free(block_id)

    def allocate(self, length: int):
        for _ in range(length):
            self._allocate_one()

    # override
    def get_this_context_len(self) -> int:
        return len(self.token_kv_block_ids)  # token len

    # override
    def get_last_token_id(self) -> int:
        return self.token_ids[-1]

    # override
    def push_token_id(self, token_id: int):
        self.token_ids.append(token_id)

    def get_context_block_ids(self) -> List[int]:
        """Return the context block ids."""

        parent_block_ids = (
            self.parent_context.get_context_block_ids() if self.parent_context else []
        )
        return parent_block_ids + self.token_kv_block_ids

    def get_context_slot_ids(self) -> List[int]:
        """Return the context slot (block + offset) ids."""

        parent_slot_ids = (
            self.parent_context.get_context_slot_ids() if self.parent_context else []
        )
        return parent_slot_ids + self.token_kv_slot_ids

    def get_last_hidden_state(self) -> torch.Tensor:
        """Return the last hidden state."""

        if len(self.token_ids) == 0:
            assert (
                self.parent_context is not None
            ), "The parent context should not be None if this is an empty context."
            return self.parent_context.get_last_hidden_state()

        return self.last_hidden_state
