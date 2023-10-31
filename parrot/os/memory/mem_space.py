from typing import Dict, List

from parrot.protocol.layer_apis import free_context
from parrot.utils import get_logger, RecyclePool
from parrot.constants import CONTEXT_POOL_SIZE, NONE_CONTEXT_ID
from parrot.exceptions import parrot_assert, ParrotOSInteralError

from ..engine import ExecutionEngine
from .context import Context
from ..process.thread import Thread, PrefixMode

logger = get_logger("Memory")


class MemorySpace:
    """A MemorySpace is a set of contexts."""

    def __init__(self):
        self.contexts: Dict[int, Context] = {}
        self.pool = RecyclePool(CONTEXT_POOL_SIZE)

        # (prefix_text, engine_id) -> prefix_context
        self.prefix_cache: Dict[List, Context] = {}
        self._prefix_cache_reversed: Dict[int, List] = {}  # for free context

        # context_id -> List of contexts
        self.process_memory: Dict[int, List[Context]] = {}

        # context_id -> ref_counter
        self.ref_counter: Dict[int, int] = {}

    # ---------- Basic Context Operation ----------

    def _new_context(self, engine: ExecutionEngine) -> Context:
        context_id = self.pool.allocate()
        context = Context(context_id=context_id, engine=engine)
        self.contexts[context_id] = context
        logger.debug(f"Context created: {context_id}")
        return context

    def _fork_context(self, parent_context: Context) -> Context:
        context_id = self.pool.allocate()
        engine = parent_context.engine
        context = Context(
            context_id=context_id,
            engine=engine,
            parent_context=parent_context,
        )
        self.contexts[context_id] = context
        logger.debug(
            f"Context created: {context_id} (Fork from {parent_context.context_id})"
        )
        return context

    def _free_context(self, context: Context):
        """Destruct the context. If we call this function, the context obj should not be used
        anymore."""

        context_id = context.context_id
        parrot_assert(
            context_id in self.ref_counter, "Context should have ref_counter."
        )
        self.ref_counter[context_id] -= 1

        if self.ref_counter[context_id] > 0:
            return

        try:
            engine = context.engine
            resp = free_context(
                http_addr=engine.http_address,
                context_id=context_id,
            )
        except BaseException as e:
            logger.error(
                f"Context: {context_id} did not free correctly: {type(e)}, {e}."
            )
            raise ParrotOSInteralError(e)
        else:
            logger.info(
                f"Context: {context_id} freed. Freed tokens: {resp.num_freed_tokens}"
            )

        self.contexts.pop(context_id)
        self.pool.free(context_id)
        if context_id in self._prefix_cache_reversed:
            prefix_key = self._prefix_cache_reversed.pop(context_id)
            self.prefix_cache.pop(prefix_key)

    def _add_ref_counter(self, context_id: int):
        if context_id not in self.ref_counter:
            self.ref_counter[context_id] = 0
        self.ref_counter[context_id] += 1

    # ---------- Memory Management ----------

    def new_memory_space(self, pid: int):
        """Create a new memory space for a process."""
        self.process_memory[pid] = []

    def free_memory_space(self, pid: int):
        """Free the memory space for a process."""
        parrot_assert(pid in self.process_memory, "Process should have memory space.")

        # Free context in topo logical order
        while len(self.process_memory[pid]) > 0:
            to_remove = set(self.process_memory[pid])
            for context in self.process_memory[pid]:
                if context.parent_context in to_remove:
                    to_remove.remove(context.parent_context)

            for context in to_remove:
                self._free_context(context)
                self.process_memory[pid].remove(context)

        self.process_memory.pop(pid)

    def free_thread_memory(self, thread: Thread):
        """Free the memory space for a thread."""
        pid = thread.process.pid
        parrot_assert(pid in self.process_memory, "Process should have memory space.")
        parrot_assert(
            self.ref_counter[thread.ctx.context_id] == 1,
            "Context should be monopolized by this thread.",
        )
        self.process_memory[pid].remove(thread.ctx)
        self._free_context(thread.ctx)

    def profile_process_memory(self, pid: int) -> float:
        """Profile the memory usage of a process."""
        parrot_assert(pid in self.process_memory, "Process should have memory space.")

        mem_used: float = 0.0
        for context in self.process_memory[pid]:
            mem_used += context.memory_usage
        return mem_used

    def set_thread_ctx(self, thread: Thread):
        """Initialize the context for a thread.
        It will first try to find a cached prefix and fork a context from it.
        If no cached prefix found:
        - If the function is marked as "cache_prefix", it will create a new context for prefix;
        - Otherwise, the whole function will be executed in the same new context.
        """
        parrot_assert(
            thread.dispatched, "Thread should be dispatched before getting context."
        )
        pid = thread.process.pid
        parrot_assert(pid in self.process_memory, "Process should have memory space.")

        # Stateful call:
        if thread.context_id_exists:
            thread.prefix_mode = PrefixMode.SAME_CTX  # Fill in the same context
            thread.ctx = self.contexts[thread.context_id]
            return

        # Not stateful call: handling prefix
        if thread.call.func.metadata.cache_prefix:
            # Try to find a cached prefix
            prefix_key = (thread.call.func.prefix.text, thread.engine.engine_id)

            if prefix_key not in self.prefix_cache:
                # No cached prefix found, create a new context
                prefix_context = self._new_context(thread.engine)
                self.prefix_cache[prefix_key] = prefix_context
                self._prefix_cache_reversed[prefix_context.context_id] = prefix_key
                thread.prefix_mode = PrefixMode.DIFF_CTX  # Fill in different context
                self._add_ref_counter(prefix_context.context_id)
                self.process_memory[pid].append(prefix_context)
            else:
                prefix_context = self.prefix_cache[prefix_key]
                logger.debug(f"Prefix cache hit! Context: {prefix_context.context_id}")
                thread.prefix_mode = PrefixMode.SKIP  # Skip the prefix fill

            new_context = self._fork_context(self.prefix_cache[prefix_key])
        else:
            new_context = self._new_context(thread.engine)
            thread.prefix_mode = PrefixMode.SAME_CTX  # Fill in the same context

        self._add_ref_counter(new_context.context_id)
        self.process_memory[pid].append(new_context)

        thread.ctx = new_context
