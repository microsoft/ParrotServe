from typing import Optional, List
from queue import Queue

from .nodes import Job, FillJob, GenerationJob, JobStatus
from ..orchestration.context import Context
from ..orchestration.engine import ExecutionEngine
from ..program.function import Promise
from ..utils import RecyclePool, get_logger
from ..protocol import fill, generate, SamplingParams
from ..protocol import free_context


session_id_manager = RecyclePool(4096)


PIPELINE_SEND_NUM = 32
PIPELINE_END_TOKEN_ID = -1


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

        # NOTE(chaofan): now we use a fixed sampling_params for all sessions
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
        )

    def __del__(self):
        # print("Session deleted.")
        session_id_manager.free(self.session_id)

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
                f"Context: {self.context.context_id} freed. Freed tokens: {resp.free_tokens_num}"
            )

    async def session_coro(self):
        while not self.job_queue.empty():
            job = self.job_queue.get()
            job.status = JobStatus.RUNNING

            logger.debug(f"Execute job: {job}")

            if isinstance(job, GenerationJob):
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
                    async for token_id in generator:
                        # Routing to pipes
                        for consumer in job.output_holder.consumers:
                            consumer.pipe.put_nowait(token_id)
                        job.output_holder.token_ids.append(token_id)

                    # Send end signals
                    for consumer in job.output_holder.consumers:
                        consumer.pipe.put_nowait(PIPELINE_END_TOKEN_ID)

                    job.output_holder.ready_event.set()
                    job.output_holder.sync_to_placeholder()
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
                        # TODO(chaofan): Merge continuous fill jobs.
                        resp = await fill(
                            self.engine.http_address,
                            session_id=self.session_id,
                            token_ids=job.input_holder.token_ids,
                            context_id=self.context.context_id,
                            # We don't fork new context. Hence parent_context_id=-1
                        )
                    else:
                        # Streaming input. Pipeling filling.
                        cur_batch: List[int] = []

                        # Fill the tokens per batch
                        while True:
                            token_id = await job.input_holder.pipe.get()

                            if token_id != PIPELINE_END_TOKEN_ID:
                                cur_batch.append(token_id)
                                job.input_holder.token_ids.append(token_id)
                            if (
                                len(cur_batch) >= PIPELINE_SEND_NUM
                                or token_id == PIPELINE_END_TOKEN_ID
                            ):
                                resp = await fill(
                                    self.engine.http_address,
                                    session_id=self.session_id,
                                    token_ids=cur_batch,
                                    context_id=self.context.context_id,
                                    # We don't fork new context. Hence parent_context_id=-1
                                )
                                assert resp.filled_tokens_num == len(
                                    cur_batch
                                ), "Fill failed: not all tokens are filled."
                                cur_batch = []

                            if token_id == PIPELINE_END_TOKEN_ID:
                                break
                except BaseException as e:
                    # TODO(chaofan): Better error handling. Current solution (abort session) will
                    # block other sessions.
                    logger.error(f"Execute fill job error: {e}")
                    break
