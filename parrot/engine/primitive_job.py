# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional
from asyncio import Event, Queue as AsyncQueue

from parrot.protocol.sampling_config import SamplingConfig

from .context.low_level_context import LowLevelContext


class PrimitiveJob:
    """Base class for all backend jobs."""

    def __init__(
        self,
        pid: int,
        tid: int,
        context_id: int,
        parent_context_id: int,
        end_flag: bool,
    ) -> None:
        self.pid = pid
        self.tid = tid
        self.end_flag = end_flag
        self.context_id = context_id
        self.parent_context_id = parent_context_id
        self.context: Optional[LowLevelContext] = None
        self.finish_event = Event()

        self.start_time: float = -1
        self.end_time: float = -1


class Fill(PrimitiveJob):
    """Fill primitive is corresponding to the `prefill` stage in LLM.

    Its mission is to fill the KV cache in the execution engine, extending the context
    using the input tokens.
    """

    def __init__(
        self,
        pid: int,
        tid: int,
        context_id: int,
        parent_context_id: int,
        end_flag: bool = False,
        token_ids: Optional[List[int]] = None,
        text: Optional[str] = None,
    ) -> None:
        super().__init__(pid, tid, context_id, parent_context_id, end_flag)
        self.token_ids = token_ids
        self.text = text

    def __repr__(self) -> str:
        return (
            f"Fill(pid={self.pid}, "
            f"tid={self.tid}, "
            f"context_id={self.context_id}, "
            f"parent_context_id={self.parent_context_id})"
        )


class Generate(PrimitiveJob):
    """Generate primitive is corresponding to the `decode` stage in LLM.

    Its mission is to generate the output tokens based on certain context.
    """

    def __init__(
        self,
        pid: int,
        tid: int,
        context_id: int,
        parent_context_id: int,
        sampling_config: SamplingConfig,
        end_flag: bool = False,
    ) -> None:
        super().__init__(pid, tid, context_id, parent_context_id, end_flag)
        self.sampling_config = sampling_config
        self.output_queue: AsyncQueue[int] = AsyncQueue()  # For token streaming
        self.gen_text = ""  # For text generation
        self.gen_length = 0

    def __repr__(self) -> str:
        return (
            f"Generate(pid={self.pid}, "
            f"tid={self.tid}, "
            f"context_id={self.context_id}, "
            f"parent_context_id={self.parent_context_id})"
        )

    # The following methods are used in the token-level context.

    def put_token(self, token_id: int) -> None:
        self.output_queue.put_nowait(token_id)

        # This requires the context to be token-level.
        self.context.push_token_id(token_id)

        self.gen_length += 1

    def check_stop(self) -> bool:
        # This requires the context to be token-level.
        token_id = self.context.get_last_token_id()
        return (
            token_id in self.sampling_config.stop_token_ids
            or self.gen_length >= self.sampling_config.max_gen_length
            # Or other stop conditions
        )

    async def generator(self):
        """Async generator for streaming."""

        while True:
            token_id = await self.output_queue.get()
            # NOTE(chaofan): We don't put the stop token into the output queue.
            if self.check_stop():
                break
            yield token_id.to_bytes(4, "big")  # streaming
