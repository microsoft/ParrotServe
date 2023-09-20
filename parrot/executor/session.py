from typing import Optional
from queue import Queue
import time

from .primitives import PrimitiveJob, Fill, Generation, JobStatus
from .tokens_holder import TokensHolder
from ..orchestration.context import Context
from ..orchestration.engine import ExecutionEngine
from ..program.function import Promise
from ..protocol import fill, generate, SamplingParams
from ..protocol import free_context
from ..utils import RecyclePool, get_logger, run_coroutine_in_loop
from ..constants import RECYCLE_POOL_SIZE, STREAMING_END_TOKEN_ID


logger = get_logger("Session")


async def detokenize_coroutine(holder: TokensHolder):
    assert holder.producer is not None, "Producer should be set."
    prev_last_token = None
    async for chunk in holder.producer.detokenize_pipe.generator():
        holder.sync_to_placeholder_partial(chunk, prev_last_token)
        prev_last_token = chunk[-1]
    holder.placeholder.ready_event.set()


class Session:
    """A session represents a running promise in the executor."""

    session_id_manager = RecyclePool(RECYCLE_POOL_SIZE)

    def __init__(self, promise: Promise, context: Context):
        # ---------- Basic info ----------
        self.session_id = Session.session_id_manager.allocate()
        self.promise = promise
        self.context = context

        # ---------- Attached engine ----------
        self.engine_name: Optional[str] = None
        self.engine: Optional[ExecutionEngine] = None

        # ---------- Job queue ----------
        self.job_queue: Queue[PrimitiveJob] = Queue()

        # NOTE(chaofan): now we use a fixed sampling_params for all sessions
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_gen_length=1024,
        )

    def __del__(self):
        # print("Session deleted.")
        Session.session_id_manager.free(self.session_id)

        try:
            resp = free_context(
                self.engine.http_address,
                self.context.context_id,
            )
        except BaseException as e:
            logger.warning(
                f"Context: {self.context.context_id} did not free correctly: {e}."
            )
        else:
            logger.info(
                f"Context: {self.context.context_id} freed. Freed tokens: {resp.num_freed_tokens}"
            )

    async def execute_coroutine(self):
        while not self.job_queue.empty():
            job = self.job_queue.get()
            job.status = JobStatus.RUNNING

            logger.debug(f"Session {self.session_id} Execute job: {job}")

            st = time.perf_counter_ns()

            if isinstance(job, Generation):
                try:
                    generator = generate(
                        self.engine.http_address,
                        session_id=self.session_id,
                        context_id=self.context.context_id,
                        sampling_params=self.sampling_params,
                        # We don't fork new context. Hence parent_context_id=-1
                    )

                    assert not job.output_holder.ready, "Output holder should be empty."
                    job.output_holder.token_ids = []

                    run_coroutine_in_loop(detokenize_coroutine(job.output_holder))

                    # Start streaming
                    job.output_holder.streaming_event.set()
                    async for token_id in generator:
                        job.output_holder.send_token(token_id)
                    job.output_holder.send_token(STREAMING_END_TOKEN_ID)

                    job.output_holder.ready_event.set()
                except BaseException as e:
                    # TODO(chaofan): Better error handling. Current solution (abort session) will
                    # block other sessions.
                    logger.error(f"Execute generation job error: {e}")
                    break
            elif isinstance(job, Fill):
                try:
                    await job.input_holder.streaming_event.wait()

                    if job.input_holder.ready:
                        # Has the whole data
                        # In this case, the placeholder must be synced.
                        # TODO(chaofan): Merge continuous fill jobs.
                        resp = await fill(
                            self.engine.http_address,
                            session_id=self.session_id,
                            token_ids=job.input_holder.token_ids,
                            context_id=self.context.context_id,
                            # We don't fork new context. Hence parent_context_id=-1
                        )
                        assert resp.num_filled_tokens == len(job.input_holder.token_ids)
                    else:
                        # Streaming input. Pipeling filling.
                        num_filled_tokens = 0
                        async for chunk in job.input_pipe.generator():
                            resp = await fill(
                                self.engine.http_address,
                                session_id=self.session_id,
                                token_ids=chunk,
                                context_id=self.context.context_id,
                                # We don't fork new context. Hence parent_context_id=-1
                            )
                            num_filled_tokens += resp.num_filled_tokens
                        assert num_filled_tokens == len(job.input_holder.token_ids)
                except BaseException as e:
                    # TODO(chaofan): Better error handling. Current solution (abort session) will
                    # block other sessions.
                    logger.error(f"Execute fill job error: {e}")
                    break

            ed = time.perf_counter_ns()
            logger.debug(f"Job {job} finished. Time used: {(ed - st) / 1e9} s.")
