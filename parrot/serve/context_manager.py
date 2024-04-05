# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Callable

from parrot.protocol.internal.layer_apis import free_context
from parrot.utils import get_logger, RecyclePool
from parrot.constants import NONE_CONTEXT_ID
from parrot.exceptions import parrot_assert, ParrotCoreInternalError

from parrot.serve.graph import CompletionChain
from parrot.serve.backend_repr import Context, ExecutionEngine
from parrot.serve.scheduler import CompletionTask


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

    def get_cached_prefix_context(self, prefix_hash: str) -> int:
        """Get the context id of a prefix from the cache.

        Args:
            prefix_hash: The hash of the prefix.

        Returns:
            The context id of the prefix. If the prefix is not in the cache, return NONE_CONTEXT_ID.
        """

        return self.prefix_cache.get(prefix_hash, NONE_CONTEXT_ID)

    def cache_prefix_context(self, prefix_hash: str, context_id: int) -> None:
        """Cache contexts of the prefix.

        Args:
            prefix_hash: The hash of the prefix.
            context_id: The context id of the prefix.
        """

        parrot_assert(
            prefix_hash not in self.prefix_cache, "Prefix should not be cached."
        )
        self.prefix_cache[prefix_hash] = context_id
        self.prefix_cache_reversed[context_id] = prefix_hash

    def remove_context_id(self, context_id: int):
        """Remove the context id of a prefix."""

        if context_id not in self.prefix_cache_reversed:
            prefix_hash = self.prefix_cache_reversed[context_id]
            self.prefix_cache.pop(prefix_hash)
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

        self.context_id_pool = RecyclePool("Context pool")

        # engine_id -> PrefixCache
        self.prefix_caches: Dict[int, PrefixCache] = {}

    # ---------- Basic Context Operation ----------

    def _new_context(self, engine: ExecutionEngine) -> Context:
        context_id = self.context_id_pool.allocate()

        # NOTE(chaofan): Context created here is not forked from any parent context by default.
        # For creating-and-forking a context, check _fork_context.
        context = Context(context_id=context_id, engine=engine)

        self.contexts[context_id] = context
        self._add_ref_counter(context)

        logger.debug(f"Context created: {context_id}")
        return context

    def _fork_context(self, parent_context: Context) -> Context:
        context_id = self.context_id_pool.allocate()
        # NOTE(chaofan): The engine of new context is the same as the parent context.
        engine = parent_context.engine

        # NOTE(chaofan): We don't need to add ref_counter for the parent context.
        # Check the logic of set_task_ctx.
        context = Context(
            context_id=context_id,
            engine=engine,
            parent_context=parent_context,
        )

        self.contexts[context_id] = context
        self._add_ref_counter(context)

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
            raise ParrotCoreInternalError(e)
        else:
            logger.debug(
                f"Context: {context_id} freed. Freed tokens: {resp.context_len}"
            )

        self.contexts.pop(context_id)
        self.context_id_pool.free(context_id)

    def _add_ref_counter(self, context: Context):
        context_id = context.context_id
        if context_id not in self.context_ref_counter:
            self.context_ref_counter[context_id] = 0
        self.context_ref_counter[context_id] += 1

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

    def set_task_ctx(self, task: CompletionTask) -> None:
        """Initialize the contexts for a CompletionTask.

        For every node,
        1. Check whether the prefix is already cached. If there is a cached context, use it.
        2. If the prefix is not cached, create a new context and cache it.
        """

        parrot_assert(
            task.scheduled, "Task should be scheduled before being set context."
        )

        chain = task.chain
        prefix_cache = self.prefix_caches[task.engine.engine_id]
        prefix_hash = ""
        prefix_no_cache_flag = False

        for node in chain.iter():
            prefix_hash += str(node.sv_id)

            if not prefix_no_cache_flag:
                # If the prefix is already cached, use cached context
                context_id = prefix_cache.get_cached_prefix_context(prefix_hash)
                if context_id != NONE_CONTEXT_ID:
                    context = self.contexts[context_id]
                    self._add_ref_counter(context)
                    task.ctxs.append(context)
                    continue
                else:
                    # For succeeding nodes, the prefix couldn't be cached.
                    prefix_no_cache_flag = True

            # The prefix is not cached. Create a new context and cache it.
            # If the node is the first node in the chain, create a new context.
            if len(task.ctxs) == 0:
                context = self._new_context(task.engine)
            # If the node is not the first node in the chain, fork the context.
            else:
                context = self._fork_context(task.ctxs[-1])

            task.ctxs.append(context)
            # Cache the context, if it's the prefix.
            if not node.is_gen:
                prefix_cache.cache_prefix_context(prefix_hash, context.context_id)

    # ---------- For Scheduler ----------

    def query_prefixes_in_engines(self, task: CompletionTask) -> List[int]:
        """Query whether there are prefixes cached in some engines.

        Args:
            task: The task to query.

        Returns:
            A list of engine ids that have cached the prefixes.
            Sorted by the number of cached prefixes in descending order.
        """

        parrot_assert(not task.scheduled, "Task should not be scheduled.")

        # engine_id -> cached_prefix_num
        sort_dict = {}

        for engine_id, prefix_cache in self.prefix_caches.items():
            prefix_hash = ""
            for node in task.chain.iter():
                prefix_hash += str(node.sv_id)
                if (
                    prefix_cache.get_cached_prefix_context(prefix_hash)
                    != NONE_CONTEXT_ID
                ):
                    if engine_id not in sort_dict:
                        sort_dict[engine_id] = 0
                    sort_dict[engine_id] += 1
                else:
                    break

        return sorted(sort_dict, key=lambda x: sort_dict[x], reverse=True)

    # ---------- Profiling ----------

    def profile_session_memory(self, session_id: int) -> float:
        """Profile the memory usage of a session."""

        parrot_assert(
            session_id in self.session_contexts, "Session should have contexts."
        )

        session_ctxs = self.session_contexts[session_id]
        return sum([ctx.memory_usage for ctx in session_ctxs])

    def profile_session_tokens(self, session_id: int) -> int:
        """Profile the total number of tokens in a session."""

        parrot_assert(
            session_id in self.session_contexts, "Session should have contexts."
        )

        session_ctxs = self.session_contexts[session_id]
        return sum([ctx.tokens_num for ctx in session_ctxs])

    # ---------- Registering ----------

    def register_session_contexts(self, session_id: int):
        """Register the contexts of a session."""

        self.session_contexts[session_id] = []

    def free_session_contexts(self, session_id: int):
        """Free the contexts of a session."""

        if session_id not in self.session_contexts:
            return

        session_ctxs = self.session_contexts[session_id]
        for ctx in session_ctxs:
            self._free_context(ctx)

        self.session_contexts.pop(session_id)

    def register_engine_prefix_cache(self, engine_id: int):
        """Register the prefix cache of an engine."""

        self.prefix_caches[engine_id] = PrefixCache()

    def remove_engine_prefix_cache(self, engine_id: int):
        """Remove the prefix cache of an engine."""

        self.prefix_caches.pop(engine_id)
