# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Callable

from parrot.protocol.internal.layer_apis import free_context
from parrot.utils import get_logger, RecyclePool
from parrot.constants import NONE_CONTEXT_ID
from parrot.exceptions import parrot_assert, ParrotOSInternalError

from parrot.serve.graph import CompletionChain
from parrot.serve.backend_repr import Context, ExecutionEngine


logger = get_logger("ContextManager")


class PrefixCache:
    """PrefixCache maps a prefix hash to a context id.

    A prefix hash is a List of SemanticVariable ids.
    """

    def __init__(self):
        # prefix hash -> context id.
        self.prefix_cache: Dict[str, int] = {}

        # reversed dict for freeing context.
        self.prefix_cache_reversed: Dict[int, str] = {}

    def _hash_prefix(self, chain: CompletionChain) -> str:
        pass

    def get_context_id(self, prefix: str) -> int:
        """Get the context id of a prefix."""
        return self.prefix_cache.get(prefix, NONE_CONTEXT_ID)

    def set_context_id(self, prefix: str, context_id: int):
        """Set the context id of a prefix."""
        self.prefix_cache[prefix] = context_id
        self.prefix_cache_reversed[context_id] = prefix

    def remove_context_id(self, prefix: str):
        """Remove the context id of a prefix."""
        context_id = self.prefix_cache.pop(prefix, NONE_CONTEXT_ID)
        if context_id != NONE_CONTEXT_ID:
            self.prefix_cache_reversed.pop(context_id)


class ServeCoreContextManager:
    """Manage all contexts in the ServeLayer.

    Note that this class is global (id pool is global), so each context_id is unique in all engines.
    """

    def __init__(self):
        # context_id -> Context
        self.contexts: Dict[int, Context] = {}

        # session_id -> List of Context
        self.session_contexts: Dict[int, List[Context]] = {}

        # context_id -> ref_counter
        # Ref counter increases when the context is used.
        # And decreases when the context is freed.

        # We track counter in ContextManager instead of Context, for putting the logic of
        # new and free context in one place.
        self.context_ref_counter: Dict[int, int] = {}

        self.id_pool = RecyclePool("Context pool")

        self.prefix_cache = PrefixCache()

    # ---------- Basic Context Operation ----------

    def _new_context(self, engine: ExecutionEngine) -> Context:
        context_id = self.id_pool.allocate()
        # NOTE(chaofan): Context created here is not forked from any parent context by default.
        # For creating-and-forking a context, check _fork_context.
        context = Context(context_id=context_id, engine=engine)
        self.contexts[context_id] = context
        logger.debug(f"Context created: {context_id}")
        return context

    def _fork_context(self, parent_context: Context) -> Context:
        context_id = self.id_pool.allocate()
        # NOTE(chaofan): The engine of new context is the same as the parent context.
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
        context_id = context.context_id
        parrot_assert(
            context_id in self.context_ref_counter, "Context should have ref_counter."
        )
        self.context_ref_counter[context_id] -= 1

        if self.context_ref_counter[context_id] > 0:
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

    # ---------- Memory Management Public Methods ----------

    def free_context(self, context: Context) -> None:
        """Free the context and return the number of freed tokens.

        If we call this function, the context obj should not be used anymore.
        """

        parrot_assert(
            context.context_id in self.contexts,
            "Context should be in the context pool.",
        )
        self._free_context(context)

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

    def profile_session_memory(self, pid: int) -> float:
        """Profile the memory usage of a session."""
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
