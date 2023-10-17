from typing import Dict

from parrot.protocol.layer_apis import free_context
from parrot.utils import get_logger, RecyclePool
from parrot.constants import CONTEXT_POOL_SIZE

from .context import Context


logger = get_logger("Memory")


class MemorySpace:
    """A MemorySpace is a set of contexts."""

    def __init__(self):
        self.contexts: Dict[int, Context] = {}
        self.pool = RecyclePool(CONTEXT_POOL_SIZE)

    def new_context(self, pid: int) -> Context:
        context_id = self.pool.allocate()
        context = Context(context_id=context_id, pid=pid)
        self.contexts[context_id] = context
        logger.debug(f"Context created: {context_id}")
        return context

    def fork_context(self, pid: int, parent_context: Context) -> Context:
        context_id = self.pool.allocate()
        context = Context(context_id=context_id, pid=pid, parent_context=parent_context)
        self.contexts[context_id] = context
        logger.debug(
            f"Context created: {context_id} (Fork from {parent_context.context_id}))"
        )
        return context

    def free_context(self, context: Context):
        """Destruct the context. If we call this function, the context obj should not be used
        anymore."""

        for engine in context.cached_engines:
            try:
                resp = free_context(
                    http_addr=engine.http_address,
                    client_id=engine.client_id,
                    context_id=context.context_id,
                )
            except BaseException as e:
                logger.error(
                    f"Context: {context.context_id} did not free correctly: {type(e)}, {e}."
                )
            else:
                logger.info(
                    f"Context: {context.context_id} freed. Freed tokens: {resp.num_freed_tokens}"
                )

        self.contexts.pop(context.context_id)
        self.pool.free(context.context_id)
