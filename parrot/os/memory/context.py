# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional
from asyncio import Event

from parrot.constants import NONE_CONTEXT_ID

from ..engine import ExecutionEngine


class Context:
    """Context represents a part of sequences cached in one single engine.

    If B wants to continue generating based on A's context, the lifecycle is:
    - B forks a context based on A's context.
    - B generates tokens in this context.
    - When B's job finish, we free the memory taken by B's context. This will not
      affect A's context.
    """

    def __init__(
        self,
        context_id: int,
        engine: ExecutionEngine,
        parent_context: Optional["Context"] = None,
    ):
        self.context_id = context_id
        self.engine = engine
        self.parent_context = parent_context
        self.token_nums = 0
        self.prefix_ready_event = Event()

    @property
    def parent_context_id(self) -> int:
        return (
            self.parent_context.context_id
            if self.parent_context is not None
            else NONE_CONTEXT_ID
        )

    @property
    def memory_usage(self) -> float:
        memory_per_token = (
            self.engine.runtime_info.cache_mem
            / self.engine.runtime_info.num_cached_tokens
            if self.engine.runtime_info.num_cached_tokens > 0
            else 0
        )

        return memory_per_token * self.token_nums

    @property
    def engine_url(self) -> str:
        return self.engine.http_address
