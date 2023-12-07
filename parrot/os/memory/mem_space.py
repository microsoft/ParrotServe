# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Callable

from parrot.protocol.layer_apis import free_context
from parrot.utils import get_logger, RecyclePool
from parrot.constants import CONTEXT_POOL_SIZE, NONE_CONTEXT_ID
from parrot.exceptions import parrot_assert, ParrotOSInternalError

from ..engine import ExecutionEngine
from .context import Context
from ..process.thread import Thread, PrefixMode

logger = get_logger("Memory")


class ProcessMemorySpace:
    """Memory space for a process."""

    def __init__(self, pid: int, add_callback: Callable, free_callback: Callable):
        self.pid = pid

        self._contexts: List[Context] = []
        self._stateful_context_table: Dict[str, Context] = {}

        self._add_callback = add_callback
        self._free_callback = free_callback

        logger.info(f"Process (pid={pid}) memory space created.")

    def add_context(self, context: Context):
        """Add a context to the process memory space. The callback will add the ref counter."""
        self._contexts.append(context)
        self._add_callback(context)

    def remove_context(self, context: Context):
        """Remove a context from the process memory space. The callback will remove
        the context in the global."""
        self._contexts.remove(context)
        self._free_callback(context)

    def free_space(self):
        """Free the memory space for a process."""

        # Free context in topo logical order
        while len(self._contexts) > 0:
            to_remove = set(self._contexts)
            for context in self._contexts:
                if context.parent_context in to_remove:
                    to_remove.remove(context.parent_context)

            for context in to_remove:
                self.remove_context(context)

        logger.info(f"Process (pid={self.pid}) memory space freed.")

    def get_mem_usage(self) -> float:
        """Get the memory usage of the process."""
        return sum([context.memory_usage for context in self._contexts])

    def get_total_tokens(self) -> int:
        """Get the total number of tokens in the process memory space."""
        return sum([context.token_nums for context in self._contexts])

    def get_state_context_id(self, func_name: str) -> int:
        """Get the context id of a stateful function."""

        context_id = self._stateful_context_table.pop(func_name, NONE_CONTEXT_ID)
        if context_id != NONE_CONTEXT_ID:
            logger.debug(
                f"Get context (conext_id={context_id}) for stateful function {func_name}."
            )
        return context_id

    def set_state_context_id(self, func_name: str, context_id: int):
        """Update the context id of a stateful function."""

        logger.debug(
            f"Set context (conext_id={context_id}) for stateful function {func_name}."
        )
        self._stateful_context_table[func_name] = context_id


class MemorySpace:
    """A MemorySpace is a set of contexts."""

    def __init__(self):
        self.contexts: Dict[int, Context] = {}
        self.pool = RecyclePool("Context pool", CONTEXT_POOL_SIZE)

        # (prefix_text, engine_id) -> prefix_context
        self.prefix_cache: Dict[List, Context] = {}
        self._prefix_cache_reversed: Dict[int, List] = {}  # for free context

        # context_id -> ProcessMemorySpace
        self.process_memory: Dict[int, ProcessMemorySpace] = {}

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
            raise ParrotOSInternalError(e)
        else:
            logger.debug(
                f"Context: {context_id} freed. Freed tokens: {resp.context_len}"
            )

        self.contexts.pop(context_id)
        self.pool.free(context_id)
        if context_id in self._prefix_cache_reversed:
            prefix_key = self._prefix_cache_reversed.pop(context_id)
            self.prefix_cache.pop(prefix_key)

    def _add_ref_counter(self, context: Context):
        context_id = context.context_id
        if context_id not in self.ref_counter:
            self.ref_counter[context_id] = 0
        self.ref_counter[context_id] += 1

    # ---------- Memory Management ----------

    def new_memory_space(self, pid: int):
        """Create a new memory space for a process."""
        self.process_memory[pid] = ProcessMemorySpace(
            pid=pid,
            add_callback=self._add_ref_counter,
            free_callback=self._free_context,
        )

    def free_memory_space(self, pid: int):
        """Free the memory space for a process."""
        parrot_assert(pid in self.process_memory, "Process should have memory space.")

        self.process_memory[pid].free_space()
        self.process_memory.pop(pid)

    def free_thread_memory(self, thread: Thread):
        """Free the memory space for a thread."""
        pid = thread.process.pid
        parrot_assert(pid in self.process_memory, "Process should have memory space.")
        parrot_assert(
            self.ref_counter[thread.ctx.context_id] == 1,
            "Context should be monopolized by this thread.",
        )
        self.process_memory[pid].remove_context(thread.ctx)

    def get_state_context_id(self, pid: int, func_name: StopIteration) -> int:
        """Get the context id of a stateful function."""
        parrot_assert(pid in self.process_memory, "Process should have memory space.")
        return self.process_memory[pid].get_state_context_id(func_name)

    def set_state_context_id(self, pid: int, func_name: str, context_id: int):
        """Update the context id of a stateful function."""
        parrot_assert(pid in self.process_memory, "Process should have memory space.")
        self.process_memory[pid].set_state_context_id(func_name, context_id)

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
                self.process_memory[pid].add_context(prefix_context)
            else:
                prefix_context = self.prefix_cache[prefix_key]
                logger.debug(
                    f"Thread {thread.unique_id}: Prefix cache hit! Using prefix ontext: {prefix_context.context_id}"
                )
                thread.prefix_mode = PrefixMode.SKIP  # Skip the prefix fill

            new_context = self._fork_context(self.prefix_cache[prefix_key])
        else:
            new_context = self._new_context(thread.engine)
            thread.prefix_mode = PrefixMode.SAME_CTX  # Fill in the same context

        self.process_memory[pid].add_context(new_context)

        thread.ctx = new_context
        thread.context_id = new_context.context_id
    
    # ---------- Profiling & Info ----------

    def profile_process_memory(self, pid: int) -> float:
        """Profile the memory usage of a process."""
        parrot_assert(pid in self.process_memory, "Process should have memory space.")
        return self.process_memory[pid].get_mem_usage()

    def profile_process_tokens(self, pid: int) -> int:
        """Profile the total number of tokens in a process."""
        parrot_assert(pid in self.process_memory, "Process should have memory space.")
        return self.process_memory[pid].get_total_tokens()
    
    def get_engines_with_ctx(self, ctx: str) -> List[int]:
        """Get the engines that have a context."""

        ret = []
        for key in self.prefix_cache:
            if key[0] == ctx:
                ret.append(key[1])
        return ret
        
