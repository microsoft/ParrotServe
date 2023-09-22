from typing import List, Optional
from asyncio import Event, Queue as AsyncQueue

from .mem import KVContext
from ..protocol.sampling_params import SamplingParams
from ..constants import STREAMING_END_TOKEN_ID


class PrimitiveJob:
    """Base class for all backend jobs."""

    def __init__(
        self,
        session_id: int,
        context_id: int,
        parent_context_id: int,
    ) -> None:
        self.session_id = session_id
        self.context_id = context_id
        self.parent_context_id = parent_context_id
        self.context: Optional[KVContext] = None
        self.finish_event = Event()


class Fill(PrimitiveJob):
    """Fill primitive is corresponding to the `prefill` stage in LLM.

    Its mission is to fill the KV cache in the execution engine, extending the context
    using the input tokens.
    """

    def __init__(
        self,
        session_id: int,
        context_id: int,
        parent_context_id: int,
        token_ids: List[int],
    ) -> None:
        super().__init__(session_id, context_id, parent_context_id)
        self.token_ids = token_ids

    def __repr__(self) -> str:
        return (
            f"Fill(session_id={self.session_id}, context_id={self.context_id}, "
            f"parent_context_id={self.parent_context_id})"
        )


class Generation(PrimitiveJob):
    """Generation primitive is corresponding to the `decode` stage in LLM.

    Its mission is to generate the output tokens based on certain context.
    """

    def __init__(
        self,
        session_id: int,
        context_id: int,
        parent_context_id: int,
        sampling_params: SamplingParams,
    ) -> None:
        super().__init__(session_id, context_id, parent_context_id)
        self.sampling_params = sampling_params
        self.output_queue: AsyncQueue[int] = AsyncQueue()
        self.gen_length = 0

    def __repr__(self) -> str:
        return (
            f"Generation(session_id={self.session_id}, context_id={self.context_id}, "
            f"parent_context_id={self.parent_context_id})"
        )

    def put_token(self, token_id: int) -> None:
        self.output_queue.put_nowait(token_id)
        self.gen_length += 1

    def check_stop(self) -> bool:
        token_id = self.context.get_last_token_id()
        return (
            token_id in self.sampling_params.stop_token_ids
            or self.gen_length >= self.sampling_params.max_gen_length
            # Or other stop conditions
        )

    async def generator(self):
        """Async generator for streaming."""

        while True:
            token_id = await self.output_queue.get()
            if self.check_stop():
                break
            yield token_id.to_bytes(4, "big")  # streaming
