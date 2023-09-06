from typing import Optional
from queue import Queue
from asyncio import Event
from aiohttp import ClientSession

from .nodes import Job, FillJob, GenerationJob
from ..orchestration.context import Context
from ..orchestration.engine import ExecutionEngine
from ..program.function import Promise
from ..utils import RecyclePool, get_logger
from ..protocol import fill, generate, SamplingParams


session_id_manager = RecyclePool(4096)


logger = get_logger("Session")


class Session:
    """A session represents a running promise in the executor."""

    def __init__(self, promise: Promise, context: Context, detokenize_queue: Queue):
        # ---------- Basic info ----------
        self.session_id = session_id_manager.allocate()
        self.promise = promise
        self.context = context
        self.detokenize_queue = detokenize_queue

        # ---------- Attached engine ----------
        self.engine_name: Optional[str] = None
        self.engine: Optional[ExecutionEngine] = None

        # ---------- Job queue ----------
        self.job_queue: Queue[Job] = Queue()

        # ---------- aiohttp session ----------
        self.client_session: ClientSession = ClientSession()

        # NOTE(chaofan): now we use a fixed sampling_params for all sessions
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
        )

    def __del__(self):
        session_id_manager.free(self.session_id)

    async def session_coro(self):
        while self.job_queue.not_empty:
            job = self.job_queue.get()
            if isinstance(job, GenerationJob):
                try:
                    resp = await generate(
                        self.client_session,
                        self.engine.http_address,
                        self.context.context_id,
                        self.sampling_params,
                    )
                    job.output_holder.assign(resp.gen_tokens)
                    self.detokenize_queue.put(job.output_holder)
                except BaseException as e:
                    # TODO(chaofan): Better error handling. Current solution (abort session) will
                    # block other sessions.
                    logger.error(f"Execute generation job error: {e}")
                    break
            elif isinstance(job, FillJob):
                try:
                    await job.input_holder.ready_event.wait()
                    resp = await fill(
                        self.client_session,
                        self.engine.http_address,
                        self.context.context_id,
                        job.input_holder.token_ids,
                    )
                    # TODO(chaofan): Pipeline filling
                except BaseException as e:
                    # TODO(chaofan): Better error handling. Current solution (abort session) will
                    # block other sessions.
                    logger.error(f"Execute fill job error: {e}")
                    break
