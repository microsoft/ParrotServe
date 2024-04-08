# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List

from parrot.protocol.internal.layer_apis import free_context
from parrot.utils import get_logger, RecyclePool
from parrot.constants import NONE_CONTEXT_ID
from parrot.exceptions import parrot_assert, ParrotCoreInternalError

from parrot.serve.backend_repr import Context, ExecutionEngine
from parrot.serve.scheduler import CompletionTask


logger = get_logger("ContextManager")


_PREFIX_HASH_BRACKET_LEFT = "{{"
_PREFIX_HASH_BRACKET_RIGHT = "}}"


class PrefixCache:
    """PrefixCache maps a prefix hash to a context id.

    A prefix hash is a List of SemanticVariable ids.

    Example:
    {{sv0}} -> Context0
    {{sv0}}{{sv1}} -> Context1
    {{sv0}}{{sv1}}{{sv2}} -> Context2
    {{sv0}}{{sv1}}{{sv3}} -> Context3
    """

    def __init__(self):
        # prefix hash -> context id.
        self._prefix_ctx_map: Dict[str, int] = {}

        # reversed dict for freeing context.
        self._prefix_ctx_map_reversed: Dict[int, str] = {}

    def get_cached_prefix_context(self, prefix_hash: str) -> int:
        """Get the context id of a prefix from the cache.

        Args:
            prefix_hash: The hash of the prefix.

        Returns:
            The context id of the prefix. If the prefix is not in the cache, return NONE_CONTEXT_ID.
        """

        return self._prefix_ctx_map.get(prefix_hash, NONE_CONTEXT_ID)

    def cache_prefix_context(self, prefix_hash: str, context_id: int) -> None:
        """Cache contexts of the prefix.

        Args:
            prefix_hash: The hash of the prefix.
            context_id: The context id of the prefix.
        """

        parrot_assert(
            prefix_hash not in self._prefix_ctx_map, "Prefix should not be cached."
        )
        self._prefix_ctx_map[prefix_hash] = context_id
        self._prefix_ctx_map_reversed[context_id] = prefix_hash

    def remove_context_id(self, context_id: int) -> None:
        """Remove the context id of a prefix."""

        if context_id not in self._prefix_ctx_map_reversed:
            prefix_hash = self._prefix_ctx_map_reversed[context_id]
            self._prefix_ctx_map.pop(prefix_hash)
            self._prefix_ctx_map_reversed.pop(context_id)


class ServeCoreContextManager:
    """Manage all contexts in the ServeLayer.

    Since Context can be forked and shared by different tasks in the same session/different sessions,
    we use a ref_counter to track the usage of the context. Normally, a Context is actually freed when
    the ref_counter decreases to 0.

    Note that this class is global (id pool is global), so each context_id is unique in all engines.
    """

    def __init__(self):
        # context_id -> Context
        self._contexts: Dict[int, Context] = {}

        # session_id -> List of Context
        self._session_contexts: Dict[int, List[Context]] = {}

        # sv_id -> List of context ids
        # Record extra ref_counters contributed by constant prefix variables.
        # If a constant prefix variable is freed, we should decrease the ref_counter of the
        # corresponding contexts.
        self._constant_prefix_contexts: Dict[str, List[Context]] = {}

        # context_id -> ref_counter
        # Ref counter increases when the context is used.
        # And decreases when the context is freed.
        # We track counter in ContextManager instead of Context, for putting the logic of
        # new and free context in one place.
        self._context_ref_counter: Dict[int, int] = {}

        self._context_id_pool = RecyclePool("Context pool")

        # engine_id -> PrefixCache
        self._prefix_caches: Dict[int, PrefixCache] = {}

    @staticmethod
    def _hash_sv_id(sv_id: str) -> str:
        return f"{_PREFIX_HASH_BRACKET_LEFT}{sv_id}{_PREFIX_HASH_BRACKET_RIGHT}"

    # ---------- Basic Context Operation ----------

    def _new_context(self, engine: ExecutionEngine) -> Context:
        context_id = self._context_id_pool.allocate()

        # NOTE(chaofan): Context created here is not forked from any parent context by default.
        # For creating-and-forking a context, check _fork_context.
        context = Context(context_id=context_id, engine=engine)

        self._contexts[context_id] = context
        self._add_ref_counter(context)

        logger.debug(f"Context created: {context_id}")
        return context

    def _fork_context(self, parent_context: Context) -> Context:
        context_id = self._context_id_pool.allocate()
        # NOTE(chaofan): The engine of new context is the same as the parent context.
        engine = parent_context.engine

        # NOTE(chaofan): We don't need to add ref_counter for the parent context.
        # Check the logic of set_task_ctx.
        context = Context(
            context_id=context_id,
            engine=engine,
            parent_context=parent_context,
        )

        self._contexts[context_id] = context
        self._add_ref_counter(context)

        logger.debug(
            f"Context created: {context_id} (Fork from {parent_context.context_id})"
        )
        return context

    def _free_context(self, context: Context) -> None:
        context_id = context.context_id
        parrot_assert(
            context_id in self._context_ref_counter, "Context should have ref_counter."
        )
        self._context_ref_counter[context_id] -= 1

        if self._context_ref_counter[context_id] > 0:
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

        # Remove context from the PrefixCache.
        prefix_cache = self._prefix_caches[engine.engine_id]
        prefix_cache.remove_context_id(context_id)

        # Remove context from the Manager.
        self._contexts.pop(context_id)
        self._context_id_pool.free(context_id)

    def _add_ref_counter(self, context: Context) -> None:
        context_id = context.context_id
        if context_id not in self._context_ref_counter:
            self._context_ref_counter[context_id] = 0
        self._context_ref_counter[context_id] += 1

    # ---------- Memory Management Public Methods ----------

    def free_context(self, context: Context) -> None:
        """Free the context and return the number of freed tokens.

        If we call this function, the context obj should not be used anymore.
        """

        parrot_assert(
            context.context_id in self._contexts,
            "Context should be in the context pool.",
        )
        self._free_context(context)

    def set_task_contexts(self, task: CompletionTask) -> None:
        """Initialize the contexts for a CompletionTask.

        For every node,
        1. Check whether the prefix is already cached. If there is a cached context, use it.
        2. If the prefix is not cached, create a new context and cache it.
        """

        parrot_assert(task.chain.sv_created, "SVs are not created yet.")
        parrot_assert(
            task._scheduled_event, "Task should be scheduled before being set context."
        )

        chain = task.chain
        prefix_cache = self._prefix_caches[task.engine.engine_id]
        prefix_hash = ""
        prefix_no_cache_flag = False

        for node in chain.iter():
            prefix_hash += self._hash_sv_id(node.sv_id)

            if not prefix_no_cache_flag:
                # If the prefix is already cached, use cached context
                context_id = prefix_cache.get_cached_prefix_context(prefix_hash)
                if context_id != NONE_CONTEXT_ID:
                    context = self._contexts[context_id]
                    self._add_ref_counter(context)
                    task.contexts.append(context)
                    continue
                else:
                    # For succeeding nodes, the prefix couldn't be cached.
                    prefix_no_cache_flag = True

            # The prefix is not cached. Create a new context and cache it.
            # If the node is the first node in the chain, create a new context.
            if len(task.contexts) == 0:
                context = self._new_context(task.engine)
                # If this is a constant prefix context, we add an extra ref_counter.
                if node.sv.is_constant_prefix:
                    if node.sv.id not in self._constant_prefix_contexts:
                        self._constant_prefix_contexts[node.sv.id] = []
                    parrot_assert(
                        context not in self._constant_prefix_contexts[node.sv.id],
                        "Context should not be in the ref map.",
                    )
                    self._constant_prefix_contexts[node.sv.id].append(context)
                    self._add_ref_counter(context)
            # If the node is not the first node in the chain, fork the context.
            else:
                context = self._fork_context(task.contexts[-1])

            task.contexts.append(context)
            # Cache the context, if it's the prefix.
            if not node.is_gen:
                prefix_cache.cache_prefix_context(prefix_hash, context.context_id)

    def free_task_contexts(self, task: CompletionTask) -> None:
        """Free the contexts of a task."""

        parrot_assert(
            task._scheduled_event,
            "Task should be scheduled before being freed context.",
        )

        for context in task.contexts:
            self._free_context(context)

    def free_constant_prefix_contexts(self, sv_id: str) -> None:
        """Free the contexts of a constant prefix variable."""

        parrot_assert(
            sv_id in self._constant_prefix_contexts,
            "Constant prefix variable should have contexts.",
        )

        for context in self._constant_prefix_contexts[sv_id]:
            self._free_context(context)

        self._constant_prefix_contexts.pop(sv_id)

    # ---------- For Scheduler ----------

    def query_prefixes_in_engines(self, task: CompletionTask) -> List[int]:
        """Query whether there are prefixes cached in some engines.

        Args:
            task: The task to query.

        Returns:
            A list of engine ids that have cached the prefixes.
            Sorted by the number of cached prefixes in descending order.
        """

        parrot_assert(not task.is_scheduled, "Task should not be scheduled.")

        # engine_id -> cached_prefix_num
        sort_dict = {}

        for engine_id, prefix_cache in self._prefix_caches.items():
            prefix_hash = ""
            for node in task.chain.iter():
                prefix_hash += self._hash_sv_id(node.sv_id)
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
            session_id in self._session_contexts, "Session should have contexts."
        )

        session_ctxs = self._session_contexts[session_id]
        return sum([ctx.memory_usage for ctx in session_ctxs])

    def profile_session_tokens(self, session_id: int) -> int:
        """Profile the total number of tokens in a session."""

        parrot_assert(
            session_id in self._session_contexts, "Session should have contexts."
        )

        session_ctxs = self._session_contexts[session_id]
        return sum([ctx.tokens_num for ctx in session_ctxs])

    # ---------- Registering ----------

    def register_session_contexts(self, session_id: int):
        """Register the contexts of a session."""

        self._session_contexts[session_id] = []

    def free_session_contexts(self, session_id: int):
        """Free the contexts of a session."""

        if session_id not in self._session_contexts:
            return

        session_ctxs = self._session_contexts[session_id]
        for ctx in session_ctxs:
            self._free_context(ctx)

        self._session_contexts.pop(session_id)

    def register_engine_prefix_cache(self, engine_id: int):
        """Register the prefix cache of an engine."""

        self._prefix_caches[engine_id] = PrefixCache()

    def remove_engine_prefix_cache(self, engine_id: int):
        """Remove the prefix cache of an engine."""

        self._prefix_caches.pop(engine_id)
