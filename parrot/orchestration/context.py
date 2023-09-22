from typing import Optional

from ..utils import RecyclePool
from ..constants import RECYCLE_POOL_SIZE, NONE_CONTEXT_ID


class Context:
    """Context represents a part of sequences cached in engines.

    If B wants to continue generating based on A's context, the lifecycle is:
    - B forks a context based on A's context.
    - B generates tokens in this context.
    - When B's job finish, we free the memory taken by B's context. This will not
      affect A's context.
    """

    context_id_manager = RecyclePool(RECYCLE_POOL_SIZE)

    def __init__(self, parent_context: Optional["Context"] = None):
        self.context_id = Context.context_id_manager.allocate()
        self.parent_context = parent_context

    def __del__(self):
        # print("Context deleted.")
        Context.context_id_manager.free(self.context_id)

    @property
    def parent_context_id(self) -> int:
        return (
            self.parent_context.context_id
            if self.parent_context is not None
            else NONE_CONTEXT_ID
        )
