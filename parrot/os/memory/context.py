from typing import Optional, Set

from parrot.constants import NONE_CONTEXT_ID
from parrot.protocol import free_context

from ..engine import ExecutionEngine


class Context:
    """Context represents a part of sequences cached in engines.

    If B wants to continue generating based on A's context, the lifecycle is:
    - B forks a context based on A's context.
    - B generates tokens in this context.
    - When B's job finish, we free the memory taken by B's context. This will not
      affect A's context.
    """

    def __init__(
        self, context_id: int, pid: int, parent_context: Optional["Context"] = None
    ):
        self.context_id = context_id
        self.pid = pid
        self.parent_context = parent_context

        # Record engines that have cached this context.
        self.cached_engines: Set[ExecutionEngine] = set()

    @property
    def parent_context_id(self) -> int:
        return (
            self.parent_context.context_id
            if self.parent_context is not None
            else NONE_CONTEXT_ID
        )
