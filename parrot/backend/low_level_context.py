from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import time

from parrot.constants import CONTEXT_EXPIRE_TIME, NONE_CONTEXT_ID
from parrot.utils import get_logger


logger = get_logger("LowLevelContext")


def get_unique_context_id(client_id: str, context_id: int) -> str:
    return f"{client_id}_{context_id}"


class LowLevelContext(ABC):
    """Base class for low-level implementation of Context."""

    def __init__(
        self,
        client_id: str,
        context_id: int,
        parent_context: Optional["LowLevelContext"],
    ):
        self.client_id = client_id
        self.context_id = context_id
        self.sub_context_ids: List[int] = []
        self.last_heartbeat_time = time.perf_counter_ns()

        # Link with parent context
        self.parent_context = parent_context
        if self.parent_context is not None:
            parent_context.sub_context_ids.append(self.context_id)

    def destruction(self):
        """Destruct the context. If we call this function, the context obj should not be used
        anymore."""

        if self.parent_context is not None:
            self.parent_context.sub_context_ids.remove(self.context_id)
        assert (
            len(self.sub_context_ids) == 0
        ), f"Sub-contexts {self.sub_context_ids[0]} should be deleted first."

    @abstractmethod
    def get_context_len(self) -> int:
        """Return the length of the context."""

    @abstractmethod
    def get_this_context_len(self) -> int:
        """Return the length of the context, without recursing into parent contexts."""

    def flush_time(self):
        """Flush the last heartbeat time."""
        self.last_heartbeat_time = time.perf_counter_ns()

    def check_expired(self) -> bool:
        """Check if the context is expired."""
        cur_time = time.perf_counter_ns()
        return cur_time - self.last_heartbeat_time > CONTEXT_EXPIRE_TIME * 1e9

    @property
    def unique_id(self) -> str:
        return get_unique_context_id(self.client_id, self.context_id)


class ContextManager:
    """Manage all low-level contexts."""

    def __init__(self):
        self._map: Dict[str, LowLevelContext] = {}

    def free_context(self, client_id: str, context_id: int) -> int:
        """Free the context and return the number of freed tokens."""

        unique_context_id = get_unique_context_id(client_id, context_id)
        if unique_context_id not in self._map:
            # raise RuntimeError(f"Context id {context_id} not found.")
            # NOTE(chaofan): There are some cases that, the context hasn't been allocated by its first
            # Fill/Generation, but it is freed because of Exception in the frontend.
            # In this case, we should just return 0.
            return 0
        context = self._map.pop(unique_context_id)
        num_freed_tokens = len(context.token_ids)
        context.destruction()
        return num_freed_tokens

    def bind_job_context(self, job: "PrimitiveJob", ctx_cls, **ctx_kwargs):
        """Set the `context` attribute of the job."""

        unique_context_id = get_unique_context_id(job.client_id, job.context_id)
        if unique_context_id not in self._map:
            # assert isinstance(job, Fill)
            if job.parent_context_id == NONE_CONTEXT_ID:
                parent_context = None
            else:
                # NOTE(chaofan): Parent context and child context must comes from
                # the same client.
                parent_context = self._map[
                    get_unique_context_id(job.client_id, job.parent_context_id)
                ]
            self._map[unique_context_id] = ctx_cls(
                job.client_id,
                job.context_id,
                parent_context,
                **ctx_kwargs,
            )
        job.context = self._map[unique_context_id]

    def flush_context_heartbeat(self, client_id: str):
        """Flush the heartbeat time of all contexts of the client."""

        for context in self._map.values():
            if context.client_id == client_id:
                context.flush_time()

    def garbage_collect(self):
        """Garbage collect the expired contexts."""

        expired_context: List[LowLevelContext] = []
        for context in self._map.values():
            if context.check_expired():
                expired_context.append(context)

        for context in expired_context:
            freed_tokens = self.free_context(context.client_id, context.context_id)
            logger.info(
                f"Garbage collect context {context.unique_id} with {freed_tokens} tokens freed."
            )

    def get_num_cached_tokens(self):
        # NOTE(chaofan): Use `get_this_context_len` instead of `get_context_len` to avoid
        # recalculation of the parent contexts.
        return sum([context.get_this_context_len() for context in self._map.values()])
