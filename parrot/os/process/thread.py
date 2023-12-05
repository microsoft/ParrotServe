# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum, auto
from typing import List, Optional
from queue import Queue
import time
import math

from parrot.program.semantic_variable import ParameterLoc
from parrot.program.function import SemanticCall
from parrot.protocol.primitive_request import Fill, Generate
from parrot.utils import get_logger, create_task_in_loop
from parrot.constants import (
    STREAMING_END_TOKEN_ID,
    FILL_NO_CHUNK,
    NONE_CONTEXT_ID,
    DETOKENIZE_CHUNK_NUM,
)
from parrot.exceptions import ParrotOSUserError, ParrotOSInternalError, parrot_assert

from .primitive_operator import *
from .placeholder import TokensHolder
from ..engine import ExecutionEngine
from ..memory.context import Context


logger = get_logger("Thread")


async def detokenizing(holder: TokensHolder):
    assert holder.producer is not None, "Producer should be set."
    prev_last_token = None
    # st = time.perf_counter_ns()
    # chunk_counter = 0
    async for chunk in holder.producer.detokenize_pipe.generator():
        # chunk_counter += 1
        holder.sync_to_placeholder_partial(chunk, prev_last_token)
        prev_last_token = chunk[-1]
    # ed = time.perf_counter_ns()
    # logger.debug(
    #     f"Detokenizing time: {(ed - st) / 1e6} ms. "
    #     f"Chunk num: {chunk_counter}, Chunk size: {DETOKENIZE_CHUNK_NUM}"
    # )
    holder.placeholder.ready_event.set()


class PrefixMode(Enum):
    """There are three cases:
    1. (SAME_CTX) The function is marked as not caching the prefix.
       In this case, we should directly create a new context.
    2. (DIFF_CTX) The function is marked as caching the prefix, but the prefix is not cached.
       In this case, we should create a new context for the prefix, and then create
       another new context for the call. The Fill operator of the prefix should be
       executed on the parent context.
    3. (SKIP) The function is marked as caching the prefix, and the prefix is cached.
       In this case, we can directly fork a new context from the cached prefix context,
       and skip the first Fill operator.
    """

    SAME_CTX = auto()
    DIFF_CTX = auto()
    SKIP = auto()


class Thread:
    """A Thread represents a running SemanticCall in the executor."""

    def __init__(
        self,
        tid: int,
        process: "Process",
        call: SemanticCall,
        context_id: int,
    ):
        # ---------- Basic info ----------
        self.process = process
        self.tid = tid
        self.call = call
        self.context_id = context_id
        self.prefix_mode = PrefixMode.SAME_CTX

        # The following resources will be set later
        self.engine: Optional[ExecutionEngine] = None
        self.ctx: Optional[Context] = None

        # ---------- Flags ----------
        self.finished_flag = False
        self.prefix_flag = False
        self.is_last_op_flag = False

        # ---------- Operators queue ----------
        self.operators: Queue[PrimitiveOperator] = Queue()

        # ---------- Fill tokens buffer ----------
        # This buffer is used to merge multiple Fill operators into one Fill primitive.
        self._fill_tokens_buffer: List[int] = []

    @property
    def unique_id(self) -> str:
        return f"{self.process.pid}_{self.tid}"

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

    @property
    def context_id_exists(self) -> bool:
        return self.context_id != NONE_CONTEXT_ID

    @property
    def is_stateful(self) -> bool:
        return self.call.context_successor is not None

    @property
    def prefix_context(self) -> Context:
        assert self.ctx is not None
        return (
            self.ctx
            if self.prefix_mode == PrefixMode.SAME_CTX
            else self.ctx.parent_context
        )

    @property
    def requests_num_upperbound(self) -> int:
        ret = 999999  # +inf
        for sv in self.call.func.body:
            if (
                isinstance(sv, ParameterLoc)
                and sv.param.dispatch_annotation is not None
            ):
                ret = min(ret, sv.param.dispatch_annotation.requests_num_upperbound)
        return ret

    def get_next_threads(self) -> List["Thread"]:
        """Get threads which take the output of this thread as input."""

        ret = set()
        for sv in self.call.func.body:
            if isinstance(sv, ParameterLoc) and sv.param.is_output:
                sv_placeholder: SVPlaceholder = self.call.bindings[sv.param.name]
                parrot_assert(
                    isinstance(sv_placeholder, SVPlaceholder),
                    f"Output loc must be a placeholder, but get {type(sv_placeholder)}",
                )

                # print(sv_placeholder, sv_placeholder.out_edges, sv_placeholder.in_edges)

                # TODO(chaofan): Here we don't consider native function edge.
                # Related works will be done in the future.
                for edge in sv_placeholder.out_edges:
                    ret.add(edge.call.thread)
        return list(ret)

    def ready_to_dispatch(self) -> bool:
        """Check whether the thread is ready to be dispatched.

        A thread is ready to be dispatched if and only if all its input placeholders are dispatched.
        """

        if self.dispatched:
            raise ParrotOSInternalError(
                f"Thread {self.unique_id} is already dispatched to engine {self.engine.engine_id}",
            )

        for sv in self.call.func.body:
            if isinstance(sv, ParameterLoc) and sv.param.is_input_loc:
                sv_placeholder = self.call.bindings[sv.param.name]
                if isinstance(sv_placeholder, SVPlaceholder):
                    # Find the producer thread
                    parrot_assert(
                        len(sv_placeholder.in_edges) <= 1,
                        f"Number of in edges must <= 1, but get multiple in-edges: {sv_placeholder.in_edges}",
                    )
                    if len(sv_placeholder.in_edges) == 1:
                        thread = sv_placeholder.in_edges[0].call.thread

                        if not thread.dispatched:
                            return False

        return True

    async def _flush_fill_tokens_buffer(self):
        buffer_len = len(self._fill_tokens_buffer)
        if buffer_len == 0:
            return

        filled_len = 0
        chunk_size = self.engine.config.fill_chunk_size

        # NOTE(chaofan): We don't chunk the tokens if it is prefix.
        if chunk_size == FILL_NO_CHUNK or not self.prefix_flag:
            chunk_size = buffer_len

        prefix_context = self.prefix_context

        chunk_num = math.ceil(buffer_len / chunk_size)

        for i in range(chunk_num):
            chunked_tokens = self._fill_tokens_buffer[
                i * chunk_size : (i + 1) * chunk_size
            ]

            end_flag = self.is_last_op_flag and i == chunk_num - 1

            if not self.prefix_flag:
                primitive = Fill(
                    pid=self.process.pid,
                    tid=self.tid,
                    context=prefix_context,
                    token_ids=chunked_tokens,
                    end_flag=end_flag,
                )
            else:
                primitive = Fill(
                    pid=self.process.pid,
                    tid=self.tid,
                    context=self.ctx,
                    token_ids=chunked_tokens,
                    end_flag=end_flag,
                )

            if not self.prefix_flag and self.prefix_mode == PrefixMode.SKIP:
                await prefix_context.prefix_ready_event.wait()
                # Skip
                filled_len += len(chunked_tokens)
            else:
                logger.debug(
                    f"Thread {self.tid} (pid={self.process.pid}) submit Fill primitive (size: {len(primitive.token_ids)})"
                )
                resp = await primitive.apost()
                filled_len += resp.filled_len

            if not self.prefix_flag:
                self.prefix_flag = True
                prefix_context.prefix_ready_event.set()

        assert (
            filled_len == buffer_len
        ), f"Not all tokens are filled. Filled: {filled_len}, total: {buffer_len}"
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
        # await op.input_holder.ready_event.wait()

        if op.input_holder.ready:
            # Has the whole data
            # In this case, the placeholder must be synced.
            self._fill_tokens_buffer.extend(op.input_holder.token_ids)
        else:
            # Streaming input. Pipeling filling.
            async for chunk in op.input_pipe.generator():
                self._fill_tokens_buffer.extend(chunk)
                await self._flush_fill_tokens_buffer()  # Flush every time

    async def _visit_token_id_placeholder_generate(
        self, op: TokenIdPlaceholderGenerate
    ):
        # Flush the buffer first.
        await self._flush_fill_tokens_buffer()

        logger.debug(
            f"Thread {self.tid} (pid={self.process.pid}) submit Generation primitive (operator: {op}, max_len: {op.sampling_config.max_gen_length})"
        )

        generator = Generate(
            pid=self.process.pid,
            tid=self.tid,
            context=self.ctx,
            end_flag=self.is_last_op_flag,
            sampling_config=op.sampling_config,
        ).astream()

        assert not op.output_holder.ready, "Output holder should be empty."
        op.output_holder.token_ids = []

        create_task_in_loop(detokenizing(op.output_holder))

        # Start streaming
        op.output_holder.streaming_event.set()
        async for token_id in generator:
            op.output_holder.send_token(token_id, put_into_holder=True)
            # asyncio.sleep(0.0001)
        op.output_holder.send_token(STREAMING_END_TOKEN_ID, put_into_holder=False)
        op.output_holder.ready_event.set()

    async def _fill_text(self, text: str):
        prefix_context = self.prefix_context

        if not self.prefix_flag:
            primitive = Fill(
                pid=self.process.pid,
                tid=self.tid,
                context=prefix_context,
                end_flag=self.is_last_op_flag,
                text=text,
            )
        else:
            primitive = Fill(
                pid=self.process.pid,
                tid=self.tid,
                context=self.ctx,
                end_flag=self.is_last_op_flag,
                text=text,
            )

        if not self.prefix_flag and self.prefix_mode == PrefixMode.SKIP:
            await prefix_context.prefix_ready_event.wait()
            # Skip
        else:
            logger.debug(
                f"Thread {self.tid} (pid={self.process.pid}) submit Fill primitive (text len: {len(text)})"
            )
            resp = await primitive.apost()

        if not self.prefix_flag:
            self.prefix_flag = True
            prefix_context.prefix_ready_event.set()

    async def _visit_text_constant_fill(self, op: TextConstantFill):
        await self._fill_text(op.text)

    async def _visit_text_placeholder_fill(self, op: TextPlaceholderFill):
        text = await op.input_holder.get()
        await self._fill_text(text)

    async def _visit_text_placeholder_generate(self, op: TextPlaceholderGenerate):
        resp = await Generate(
            pid=self.process.pid,
            tid=self.tid,
            context=self.ctx,
            end_flag=self.is_last_op_flag,
            sampling_config=op.sampling_config,
        ).apost()
        op.output_holder.set(resp.generated_text)

    async def executing(self):
        while not self.operators.empty():
            op = self.operators.get(timeout=0.5)

            if self.operators.empty():
                self.is_last_op_flag = True

            try:
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
            except Exception as e:
                # If exception happens, we should terminate the thread and the related process.
                # But other processes should not be affected.
                logger.error(f"Error when executing operator {op}: {e}")
                self.process.exception_interrupt(ParrotOSUserError(e))

        self.finished_flag = True
