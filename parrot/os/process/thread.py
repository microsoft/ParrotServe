from typing import List, Optional
from queue import Queue

from parrot.program.function import SemanticCall
from parrot.protocol.primitive_request import Fill, Generate
from parrot.utils import get_logger, create_task_in_loop
from parrot.constants import (
    STREAMING_END_TOKEN_ID,
    FILL_NO_CHUNK,
)

from .primitive_operator import *
from .placeholder import Placeholder
from ..engine import ExecutionEngine
from ..memory.context import Context


logger = get_logger("Thread")


async def detokenizing(holder: Placeholder):
    assert holder.producer is not None, "Producer should be set."
    prev_last_token = None
    async for chunk in holder.producer.detokenize_pipe.generator():
        holder.sync_to_future_partial(chunk, prev_last_token)
        prev_last_token = chunk[-1]
    holder.future.ready_event.set()


class Thread:
    """A Thread represents a running SemanticCall in the executor."""

    def __init__(
        self,
        tid: int,
        process: "Process",
        call: SemanticCall,
    ):
        # ---------- Basic info ----------
        self.process = process
        self.tid = tid
        self.call = call
        self.finished_flag = False

        # The following resources will be set later
        self.engine: Optional[ExecutionEngine] = None
        self.ctx: Optional[Context] = None

        # ---------- Operators queue ----------
        self.operators: Queue[PrimitiveOperator] = Queue()

        # ---------- Fill tokens buffer ----------
        # This buffer is used to merge multiple Fill operators into one Fill primitive.
        self._fill_tokens_buffer: List[int] = []

    @property
    def dispatched(self) -> bool:
        return self.engine is not None

    @property
    def allocated_memory(self) -> bool:
        return self.ctx is not None

    @property
    def interpreted(self) -> bool:
        return self.operators.empty() and not self.finished_flag

    @property
    def finished(self) -> bool:
        return self.finished_flag

    async def _flush_fill_tokens_buffer(self):
        buffer_len = len(self._fill_tokens_buffer)
        if buffer_len == 0:
            return

        num_filled_tokens = 0
        chunk_size = self.engine.fill_chunk_size
        if chunk_size == FILL_NO_CHUNK:
            chunk_size = buffer_len

        for i in range(buffer_len // chunk_size):
            chunked_tokens = self._fill_tokens_buffer[
                i * chunk_size : (i + 1) * chunk_size
            ]

            logger.debug(
                f"Thread {self.tid} submit Fill primitive (size: {len(chunked_tokens)})"
            )

            primitive = Fill(
                pid=self.process.pid,
                tid=self.tid,
                context_id=self.ctx.context_id,
                parent_context_id=self.ctx.parent_context_id,
                token_ids=chunked_tokens,
            )
            resp = await primitive.apost(self.engine.http_address)
            num_filled_tokens += resp.num_filled_tokens
        assert (
            num_filled_tokens == buffer_len
        ), f"Not all tokens are filled. Filled: {num_filled_tokens}, total: {buffer_len}"
        self._fill_tokens_buffer = []

    async def _visit_token_id_constant_fill(self, op: TokenIdConstantFill):
        self._fill_tokens_buffer.extend(op.token_ids)

    async def _visit_token_id_placeholder_fill(self, op: TokenIdPlaceholderFill):
        if not op.input_holder.ready:
            # Not ready. Flush the buffer first.
            await self._flush_fill_tokens_buffer()

        if op.input_holder.future.coroutine is not None:
            await op.input_holder.future._wait_content()

        # Lock unitl the input holder is streaming.
        # Then there are two cases:
        # 1. The input holder is ready. We can fill the whole data.
        # 2. The input holder is not ready. We can fill the data chunk by chunk.
        await op.input_holder.streaming_event.wait()

        if op.input_holder.ready:
            # Has the whole data
            # In this case, the placeholder must be synced.
            self._fill_tokens_buffer.extend(op.input_holder.token_ids)
        else:
            # Streaming input. Pipeling filling.
            num_filled_tokens = 0
            async for chunk in op.input_pipe.generator():
                primitive = Fill(
                    pid=self.process.pid,
                    tid=self.tid,
                    context_id=self.ctx.context_id,
                    parent_context_id=self.ctx.parent_context_id,
                    token_ids=chunk,
                )
                resp = await primitive.apost(self.engine.http_address)
                num_filled_tokens += resp.num_filled_tokens
            should_filled = len(op.input_holder.token_ids)
            assert (
                num_filled_tokens == should_filled
            ), f"Not all tokens are filled. Filled: {num_filled_tokens}, total: {should_filled}"

    async def _visit_token_id_placeholder_generate(
        self, op: TokenIdPlaceholderGenerate
    ):
        # Flush the buffer first.
        await self._flush_fill_tokens_buffer()

        logger.debug(f"Thread {self.tid} submit Generation primitive (operator: {op})")

        primitive = Generate(
            pid=self.process.pid,
            tid=self.tid,
            context_id=self.ctx.context_id,
            parent_context_id=self.ctx.parent_context_id,
            sampling_config=op.sampling_config,
        )

        generator = primitive.astream(self.engine.http_address)

        assert not op.output_holder.ready, "Output holder should be empty."
        op.output_holder.token_ids = []

        create_task_in_loop(detokenizing(op.output_holder))

        # Start streaming
        op.output_holder.streaming_event.set()
        async for token_id in generator:
            op.output_holder.send_token(token_id, put_into_holder=True)
        op.output_holder.send_token(STREAMING_END_TOKEN_ID, put_into_holder=False)
        op.output_holder.ready_event.set()

    async def _visit_text_constant_fill(self, op: TextConstantFill):
        primitive = Fill(
            pid=self.process.pid,
            tid=self.tid,
            context_id=self.ctx.context_id,
            parent_context_id=self.ctx.parent_context_id,
            text=op.text,
        )
        resp = await primitive.apost(self.engine.http_address)

    async def _visit_text_placeholder_fill(self, op: TextPlaceholderFill):
        text = await op.input_holder.get()
        primitive = Fill(
            pid=self.process.pid,
            tid=self.tid,
            context_id=self.ctx.context_id,
            parent_context_id=self.ctx.parent_context_id,
            text=text,
        )
        resp = await primitive.apost(self.engine.http_address)

    async def _visit_text_placeholder_generate(self, op: TextPlaceholderGenerate):
        primitive = Generate(
            pid=self.process.pid,
            tid=self.tid,
            context_id=self.ctx.context_id,
            parent_context_id=self.ctx.parent_context_id,
            sampling_config=op.sampling_config,
        )
        resp = await primitive.apost(self.engine.http_address)
        op.output_holder._set(resp.generated_text)

    async def executing(self):
        while not self.operators.empty():
            op = self.operators.get()

            if isinstance(op, TokenIdPlaceholderGenerate):
                self._visit_token_id_placeholder_generate(op)
            elif isinstance(op, TokenIdConstantFill):
                self._visit_token_id_constant_fill(op)
            elif isinstance(op, TokenIdPlaceholderFill):
                self._visit_token_id_placeholder_fill(op)
            elif isinstance(op, TextConstantFill):
                self._visit_text_constant_fill(op)
            elif isinstance(op, TextPlaceholderFill):
                self._visit_text_placeholder_fill(op)
            elif isinstance(op, TextPlaceholderGenerate):
                self._visit_text_placeholder_generate(op)

        self.finished_flag = True
