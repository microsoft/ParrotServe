# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict

from parrot.constants import NONE_CONTEXT_ID

from .low_level_context import LowLevelContext
from ..primitive_job import PrimitiveJob


class ContextManager:
    """Manage all low-level contexts."""

    def __init__(self):
        self._map: Dict[str, LowLevelContext] = {}

    def free_context(self, context_id: int) -> int:
        """Free the context and return the number of freed tokens.

        Return the length of the context.
        """

        if context_id not in self._map:
            # raise RuntimeError(f"Context id {context_id} not found.")
            # NOTE(chaofan): There are some cases that, the context hasn't been allocated by its first
            # Fill/Generation, but it is freed because of Exception in the frontend.
            # In this case, we should just return 0.
            return 0
        context = self._map.pop(context_id)
        context_len = context.get_this_context_len()
        context.destruction()
        return context_len

    def bind_job_context(self, job: PrimitiveJob, ctx_cls, **ctx_kwargs):
        """Set the `context` attribute of the job."""

        if job.context_id not in self._map:
            # assert isinstance(job, Fill)
            if job.parent_context_id == NONE_CONTEXT_ID:
                parent_context = None
            else:
                parent_context = self._map[job.parent_context_id]
            self._map[job.context_id] = ctx_cls(
                job.context_id,
                parent_context,
                **ctx_kwargs,
            )
        job.context = self._map[job.context_id]

    def get_num_cached_tokens(self):
        # NOTE(chaofan): Use `get_this_context_len` instead of `get_context_len` to avoid
        # recalculation of the parent contexts.
        return sum([context.get_this_context_len() for context in self._map.values()])
