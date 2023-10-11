from typing import Optional, List, Callable
from queue import Queue

from parrot.program.function import SemanticCall
from parrot.protocol.primitives import Fill, Generate
from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import RecyclePool, get_logger, create_task_in_loop
from parrot.constants import (
    RECYCLE_POOL_SIZE,
    STREAMING_END_TOKEN_ID,
    FILL_NO_CHUNK,
    NONE_CONTEXT_ID,
)

from .instructions import *
from .dataholder import DataHolder


logger = get_logger("Thread")


async def detokenizing(holder: DataHolder):
    assert holder.producer is not None, "Producer should be set."
    prev_last_token = None
    async for chunk in holder.producer.detokenize_pipe.generator():
        holder.sync_to_future_partial(chunk, prev_last_token)
        prev_last_token = chunk[-1]
    holder.future.ready_event.set()


class Thread:
    """A Thread represents a running LLMCall in the executor."""

    thread_id_manager = RecyclePool(RECYCLE_POOL_SIZE)

    def __init__(self, vm: "VirtualMachine", call: SemanticCall):
        # ---------- Basic info ----------
        self.vm = vm
        self.tid = Thread.thread_id_manager.allocate()
        self.call = call

        # ---------- Instructions queue ----------
        self.instructions: Queue[Instruction] = Queue()

        # ---------- Fill tokens buffer ----------
        # This buffer is used to merge multiple Fill instructions into one Fill primitive.
        self._fill_tokens_buffer: List[int] = []

    def __del__(self):
        # print("Thread deleted.")
        Thread.thread_id_manager.free(self.tid)

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

            resp = await afill(
                http_addr=self.engine.http_address,
                client_id=self.engine.client_id,
                Thread_id=self.tid,
                token_ids=chunked_tokens,
                context_id=self.context.context_id,
                parent_context_id=self.context.parent_context_id,
            )
            num_filled_tokens += resp.num_filled_tokens
        assert (
            num_filled_tokens == buffer_len
        ), f"Not all tokens are filled. Filled: {num_filled_tokens}, total: {buffer_len}"
        self._fill_tokens_buffer = []

    async def _visit_token_id_constant_fill(self, inst: TokenIdConstantFill):
        self._fill_tokens_buffer.extend(inst.token_ids)

    async def _visit_token_id_placeholder_fill(self, inst: TokenIdPlaceholderFill):
        if not inst.input_holder.ready:
            # Not ready. Flush the buffer first.
            await self._flush_fill_tokens_buffer()

        if inst.input_holder.future.coroutine is not None:
            await inst.input_holder.future._wait_content()

        # Lock unitl the input holder is streaming.
        # Then there are two cases:
        # 1. The input holder is ready. We can fill the whole data.
        # 2. The input holder is not ready. We can fill the data chunk by chunk.
        await inst.input_holder.streaming_event.wait()

        if inst.input_holder.ready:
            # Has the whole data
            # In this case, the placeholder must be synced.
            self._fill_tokens_buffer.extend(inst.input_holder.token_ids)
        else:
            # Streaming input. Pipeling filling.
            num_filled_tokens = 0
            async for chunk in inst.input_pipe.generator():
                primitive = Fill(
                    pid=self.vm.pid,
                    tid=self.tid,
                    context_id=NONE_CONTEXT_ID,
                    parent_context_id=NONE_CONTEXT_ID,
                    token_ids=chunk,
                )
                resp = await primitive.apost(self.vm.os_http_address)
                num_filled_tokens += resp.num_filled_tokens
            should_filled = len(inst.input_holder.token_ids)
            assert (
                num_filled_tokens == should_filled
            ), f"Not all tokens are filled. Filled: {num_filled_tokens}, total: {should_filled}"

    async def _visit_token_id_placeholder_generate(
        self, inst: TokenIdPlaceholderGenerate
    ):
        # Flush the buffer first.
        await self._flush_fill_tokens_buffer()

        logger.debug(
            f"Thread {self.tid} submit Generation primitive (instruction: {inst})"
        )

        primitive = Generate(
            pid=self.vm.pid,
            tid=self.tid,
            context_id=NONE_CONTEXT_ID,
            parent_context_id=NONE_CONTEXT_ID,
            sampling_config=inst.sampling_config,
        )

        generator = primitive.astream(self.vm.os_http_address)

        assert not inst.output_holder.ready, "Output holder should be empty."
        inst.output_holder.token_ids = []

        create_task_in_loop(detokenizing(inst.output_holder))

        # Start streaming
        inst.output_holder.streaming_event.set()
        async for token_id in generator:
            inst.output_holder.send_token(token_id, put_into_holder=True)
        inst.output_holder.send_token(STREAMING_END_TOKEN_ID, put_into_holder=False)
        inst.output_holder.ready_event.set()

    async def _visit_text_constant_fill(self, inst: TextConstantFill):
        primitive = Fill(
            pid=self.vm.pid,
            tid=self.tid,
            context_id=NONE_CONTEXT_ID,
            parent_context_id=NONE_CONTEXT_ID,
            text=inst.text,
        )
        resp = await primitive.apost(self.vm.os_http_address)

    async def _visit_text_placeholder_fill(self, inst: TextPlaceholderFill):
        text = await inst.input_holder.get()
        primitive = Fill(
            pid=self.vm.pid,
            tid=self.tid,
            context_id=NONE_CONTEXT_ID,
            parent_context_id=NONE_CONTEXT_ID,
            text=text,
        )
        resp = await primitive.apost(self.vm.os_http_address)

    async def _visit_text_placeholder_generate(self, inst: TextPlaceholderGenerate):
        primitive = Generate(
            pid=self.vm.pid,
            tid=self.tid,
            context_id=NONE_CONTEXT_ID,
            parent_context_id=NONE_CONTEXT_ID,
            sampling_config=inst.sampling_config,
        )
        resp = await primitive.apost(self.vm.os_http_address)
        inst.output_holder._set(resp.generated_text)

    async def executing(self):
        while not self.instructions.empty():
            inst = self.instructions.get()

            if isinstance(inst, TokenIdPlaceholderGenerate):
                self._visit_token_id_placeholder_generate(inst)
            elif isinstance(inst, TokenIdConstantFill):
                self._visit_token_id_constant_fill(inst)
            elif isinstance(inst, TokenIdPlaceholderFill):
                self._visit_token_id_placeholder_fill(inst)
            elif isinstance(inst, TextConstantFill):
                self._visit_text_constant_fill(inst)
            elif isinstance(inst, TextPlaceholderFill):
                self._visit_text_placeholder_fill(inst)
            elif isinstance(inst, TextPlaceholderGenerate):
                self._visit_text_placeholder_generate(inst)
