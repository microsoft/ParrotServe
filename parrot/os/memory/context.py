from typing import Optional, Set

from parrot.utils import RecyclePool, get_logger
from parrot.constants import RECYCLE_POOL_SIZE, NONE_CONTEXT_ID
from parrot.protocol import free_context

from ..engine import ExecutionEngine


logger = get_logger("Context")


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

        # Record engines that have cached this context.
        self.cached_engines: Set[ExecutionEngine] = set()

        logger.debug(f"Context created: {self.context_id}")

    def destruction(self):
        """Destruct the context. If we call this function, the context obj should not be used
        anymore."""

        for engine in self.cached_engines:
            try:
                resp = free_context(
                    http_addr=engine.http_address,
                    client_id=engine.client_id,
                    context_id=self.context_id,
                )
            except BaseException as e:
                logger.error(
                    f"Context: {self.context_id} did not free correctly: {type(e)}, {e}."
                )
            else:
                logger.info(
                    f"Context: {self.context_id} freed. Freed tokens: {resp.num_freed_tokens}"
                )

        Context.context_id_manager.free(self.context_id)

    @property
    def parent_context_id(self) -> int:
        return (
            self.parent_context.context_id
            if self.parent_context is not None
            else NONE_CONTEXT_ID
        )
