from typing import List, Optional
from asyncio import Event, Queue as AsyncQueue

from .mem import KVContext
from ..protocol.sampling_params import SamplingParams


class BackendPrimitiveJob:
    """Base class for all backend jobs."""

    def __init__(self, session_id: int, context_id: int) -> None:
        self.session_id = session_id
        self.context_id = context_id
        self.context: Optional[KVContext] = None
        self.finished = Event()


class Fill(BackendPrimitiveJob):
    def __init__(
        self,
        session_id: int,
        context_id: int,
        token_ids: List[int],
        parent_context_id: int = -1,
    ) -> None:
        super().__init__(session_id, context_id)
        self.token_ids = token_ids
        self.parent_context_id = parent_context_id

    def __repr__(self) -> str:
        return f"Fill(session_id={self.session_id}, context_id={self.context_id})"


class Generation(BackendPrimitiveJob):
    def __init__(
        self,
        session_id: int,
        context_id: int,
        sampling_params: SamplingParams,
    ) -> None:
        super().__init__(session_id, context_id)
        self.sampling_params = sampling_params
        self.output_queue: AsyncQueue[int] = AsyncQueue()

    def __repr__(self) -> str:
        return f"Generation(session_id={self.session_id}, context_id={self.context_id})"
