import asyncio
from typing import Dict

from .runner import Runner
from .scheduler import Scheduler
from .backend_jobs import BackendPrimitiveJob

from ..constants import ENGINE_LOOP_INTERVAL
from ..utils import get_logger


logger = get_logger("ExecutionEngine")


class ExecutionEngine:
    """Backend Execution Engine for Parrot."""

    def __init__(self):
        # TODO(chaofan): config these
        self.runner = Runner("facebook/opt-125m")
        self.scheduler = Scheduler(max_batch_size=256, max_tokens_sum=8192)

    def add_job(self, job: BackendPrimitiveJob):
        logger.info(f"Adding job: {job}")
        self.scheduler.add_job(job)
        self.runner.bind_job_context(job)

    def free_context(self, context_id: int) -> int:
        for job in self.scheduler.running_jobs:
            if job.context_id == context_id:
                # NOTE(chaofan): We cannot free the context when it is still running.
                raise RuntimeError("Context is still running.")

        context = self.runner.context_manager.pop(context_id)
        free_tokens_num = len(context.token_ids)
        del context
        return free_tokens_num

    def stats(self) -> Dict[str, int]:
        """Return: cached_tokens_num, cached_tokens_size. num_running_jobs."""

        cached_tokens_num = 0
        for context in self.runner.context_manager.values():
            cached_tokens_num += len(
                len(context.tokens_kv_block_id)
            )  # We don't need to count the parent context.

        cached_tokens_size = (
            cached_tokens_num
            * self.runner.model_config.hidden_size
            * self.runner.num_layers
            * 2
            / 1024
            / 1024  # MiB
        )
        num_running_jobs = len(self.scheduler.running_jobs)
        return {
            "cached_tokens_num": cached_tokens_num,
            "cached_tokens_size": cached_tokens_size,
            "num_running_jobs": num_running_jobs,
        }

    async def execute_loop(self):
        while True:
            await asyncio.sleep(ENGINE_LOOP_INTERVAL)

            if self.scheduler.empty:
                continue

            jobs = self.scheduler.schedule()

            logger.info(f"Running {len(jobs)} jobs: {jobs}")

            self.runner.run_iter(jobs)
            self.scheduler.finish()
