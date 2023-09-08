from typing import Optional, List
from queue import Queue
from asyncio import Event, BaseEventLoop
from aiohttp import ClientSession

from .nodes import Job, FillJob, GenerationJob
from ..orchestration.context import Context
from ..orchestration.engine import ExecutionEngine
from ..program.function import Promise
from ..utils import RecyclePool, get_logger
from ..protocol import fill, generate, SamplingParams


session_id_manager = RecyclePool(4096)


PIPELINE_SEND_NUM = 32


logger = get_logger("Session")


class Session:
    """A session represents a running promise in the executor."""

    def __init__(self, promise: Promise, context: Context):
        # ---------- Basic info ----------
        self.session_id = session_id_manager.allocate()
        self.promise = promise
        self.context = context

        # ---------- Attached engine ----------
        self.engine_name: Optional[str] = None
        self.engine: Optional[ExecutionEngine] = None

        # ---------- Job queue ----------
        self.job_queue: Queue[Job] = Queue()

        # ---------- aiohttp session ----------
        self.client_session: Optional[ClientSession] = None

        self.finish_event = Event()

        # NOTE(chaofan): now we use a fixed sampling_params for all sessions
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
        )

    def __del__(self):
        session_id_manager.free(self.session_id)

    def set_loop(self, loop: BaseEventLoop):
        self.client_session = ClientSession(loop=loop)

    async def session_coro(self):
        while self.job_queue.not_empty:
            job = self.job_queue.get()
            if isinstance(job, GenerationJob):
                try:
                    job.output_holder.generator = generate(
                        self.client_session,
                        self.engine.http_address,
                        self.context.context_id,
                        self.sampling_params,
                    )
                except BaseException as e:
                    # TODO(chaofan): Better error handling. Current solution (abort session) will
                    # block other sessions.
                    logger.error(f"Execute generation job error: {e}")
                    break
            elif isinstance(job, FillJob):
                try:
                    await job.input_holder.streaming_event.wait()
                    if job.input_holder.ready:
                        # Has the whole data
                        # In this case, the placeholder must be synced.
                        resp = await fill(
                            self.client_session,
                            self.engine.http_address,
                            self.context.context_id,
                            job.input_holder.token_ids,
                        )
                    else:
                        # Streaming input. Pipeling filling.
                        assert job.input_holder.generator is not None
                        cur_batch: List[int] = []

                        # Fill the tokens per batch
                        async for token_id in job.input_holder.generator:
                            cur_batch.append(token_id)
                            job.input_holder.token_ids.append(token_id)
                            if len(cur_batch) >= PIPELINE_SEND_NUM:
                                resp = await fill(
                                    self.client_session,
                                    self.engine.http_address,
                                    self.context.context_id,
                                    cur_batch,
                                )
                                assert resp.filled_tokens_num == len(
                                    cur_batch
                                ), "Fill failed: not all tokens are filled."
                                cur_batch = []

                        # Send the last batch
                        if len(cur_batch) > 0:
                            resp = await fill(
                                self.client_session,
                                self.engine.http_address,
                                self.context.context_id,
                                cur_batch,
                            )
                            assert resp.filled_tokens_num == len(
                                cur_batch
                            ), "Fill failed: not all tokens are filled."

                        job.input_holder.ready_event.set()
                        job.input_holder.sync_to_placeholder()
                except BaseException as e:
                    # TODO(chaofan): Better error handling. Current solution (abort session) will
                    # block other sessions.
                    logger.error(f"Execute fill job error: {e}")
                    break

        self.finish_event.set()
