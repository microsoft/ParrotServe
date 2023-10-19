from enum import Enum, auto
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
from .placeholder import TokensHolder
from ..engine import ExecutionEngine
from ..memory.context import Context


logger = get_logger("Thread")


async def detokenizing(holder: TokensHolder):
    assert holder.producer is not None, "Producer should be set."
    prev_last_token = None
    async for chunk in holder.producer.detokenize_pipe.generator():
        holder.sync_to_placeholder_partial(chunk, prev_last_token)
        prev_last_token = chunk[-1]
    holder.placeholder.ready_event.set()


class PrefixMode(Enum):
    """There are three cases:
    1. (NOCACHE) The function is marked as not caching the prefix.
       In this case, we should directly create a new context.
    2. (FORK) The function is marked as caching the prefix, but the prefix is not cached.
       In this case, we should create a new context for the prefix, and then create
       another new context for the call. The Fill operator of the prefix should be
       executed on the parent context.
    3. (SKIP) The function is marked as caching the prefix, and the prefix is cached.
       In this case, we can directly fork a new context from the cached prefix context,
       and skip the first Fill operator.
    """

    NOCACHE = auto()
    FORK = auto()
    SKIP = auto()


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
        self.prefix_mode = PrefixMode.NOCACHE

        # The following resources will be set later
        self.engine: Optional[ExecutionEngine] = None
        self.ctx: Optional[Context] = None

        # ---------- Flags ----------
        self.finished_flag = False
        self.prefix_flag = False

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
        chunk_size = self.engine.config.fill_chunk_size

        # NOTE(chaofan): We don't chunk the tokens if it is prefix.
        if chunk_size == FILL_NO_CHUNK or not self.prefix_flag:
            chunk_size = buffer_len

        for i in range(buffer_len // chunk_size):
            chunked_tokens = self._fill_tokens_buffer[
                i * chunk_size : (i + 1) * chunk_size
            ]

            logger.debug(
                f"Thread {self.tid} submit Fill primitive (size: {len(chunked_tokens)})"
            )

            if not self.prefix_flag:
                if self.prefix_mode == PrefixMode.SKIP:
                    # Skip the first Fill operator
                    primitive = None
                elif self.prefix_mode == PrefixMode.FORK:
                    assert self.ctx.parent_context is not None
                    primitive = Fill(
                        pid=self.process.pid,
                        tid=self.tid,
                        context=self.ctx.parent_context,
                        token_ids=chunked_tokens,
                    )
                else:
                    assert self.prefix_mode == PrefixMode.NOCACHE
                    primitive = Fill(
                        pid=self.process.pid,
                        tid=self.tid,
                        context=self.ctx,
                        token_ids=chunked_tokens,
                    )
                self.prefix_flag = True
            else:
                primitive = Fill(
                    pid=self.process.pid,
                    tid=self.tid,
                    context=self.ctx,
                    token_ids=chunked_tokens,
                )
            if primitive is not None:
                resp = await primitive.apost()
                num_filled_tokens += resp.num_filled_tokens
            else:
                # Skip
                num_filled_tokens += len(chunked_tokens)
        assert (
            num_filled_tokens == buffer_len
        ), f"Not all tokens are filled. Filled: {num_filled_tokens}, total: {buffer_len}"
        self._fill_tokens_buffer = []

    async def _visit_token_id_constant_fill(self, op: TokenIdConstantFill):
        self._fill_tokens_buffer.extend(op.token_ids)
        if not self.prefix_flag:
            # Prefix, should send a Fill primitive.
            await self._flush_fill_tokens_buffer()

    async def _visit_token_id_placeholder_fill(self, op: TokenIdPlaceholderFill):
        if not op.input_holder.ready:
            # Not ready. Flush the buffer first.
            await self._flush_fill_tokens_buffer()

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
                resp = await Fill(
                    pid=self.process.pid,
                    tid=self.tid,
                    context=self.ctx,
                    token_ids=chunk,
                ).apost()
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

        generator = Generate(
            pid=self.process.pid,
            tid=self.tid,
            context=self.ctx,
            sampling_config=op.sampling_config,
        ).astream()

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
        resp = await Fill(
            pid=self.process.pid,
            tid=self.tid,
            context=self.ctx,
            text=op.text,
        ).apost()

    async def _visit_text_placeholder_fill(self, op: TextPlaceholderFill):
        text = await op.input_holder.get()
        resp = await Fill(
            pid=self.process.pid,
            tid=self.tid,
            context=self.ctx,
            text=text,
        ).apost()

    async def _visit_text_placeholder_generate(self, op: TextPlaceholderGenerate):
        resp = await Generate(
            pid=self.process.pid,
            tid=self.tid,
            context=self.ctx,
            sampling_config=op.sampling_config,
        ).apost()
        op.output_holder.set(resp.generated_text)

    async def executing(self):
        while not self.operators.empty():
            op = self.operators.get()

            if isinstance(op, TokenIdPlaceholderGenerate):
                await self._visit_token_id_placeholder_generate(op)
            elif isinstance(op, TokenIdConstantFill):
                await self._visit_token_id_constant_fill(op)
            elif isinstance(op, TokenIdPlaceholderFill):
                await self._visit_token_id_placeholder_fill(op)
            elif isinstance(op, TextConstantFill):
                await self._visit_text_constant_fill(op)
            elif isinstance(op, TextPlaceholderFill):
                await self._visit_text_placeholder_fill(op)
            elif isinstance(op, TextPlaceholderGenerate):
                await self._visit_text_placeholder_generate(op)

        self.finished_flag = True
