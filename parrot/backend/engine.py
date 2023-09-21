import asyncio
from typing import Dict
import json

from .runner import Runner
from .scheduler import Scheduler
from .backend_jobs import BackendPrimitiveJob

from ..constants import ENGINE_LOOP_INTERVAL
from ..utils import get_logger
from .config import RunnerConfig, SchedulerConfig


logger = get_logger("ExecutionEngine")


class ExecutionEngine:
    """Backend Execution Engine for Parrot."""

    def __init__(self, engine_config_path: str):
        with open(engine_config_path) as f:
            engine_config = json.load(f)

        self.engine_name = engine_config["engine_name"]
        runner_config = RunnerConfig(**engine_config["runner"])
        scheduler_config = SchedulerConfig(**engine_config["scheduler"])
        self.runner = Runner(runner_config)
        self.scheduler = Scheduler(scheduler_config)

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
        num_freed_tokens = len(context.token_ids)
        del context
        return num_freed_tokens

    def stats(self) -> Dict[str, int]:
        """Return: num_cached_tokens, cached_tokens_size. num_running_jobs."""

        num_cached_tokens = 0
        for context in self.runner.context_manager.values():
            num_cached_tokens += len(
                context.tokens_kv_block_id
            )  # We don't need to count the parent context.

        cached_tokens_size = (
            num_cached_tokens
            # TODO(chaofan): Currently this config must be OPTConfig.
            # Support other configs in the future.
            * self.runner.hf_model_config.hidden_size
            * self.runner.hf_model_config.num_hidden_layers
            * 2
            / 1024
            / 1024  # MiB
        )
        num_running_jobs = len(self.scheduler.running_jobs)
        return {
            "num_cached_tokens": num_cached_tokens,
            "cached_tokens_size": cached_tokens_size,
            "num_running_jobs": num_running_jobs,
        }

    async def execute_loop(self):
        logger.info(f"Execution loop of engine: {self.engine_name} started.")

        while True:
            await asyncio.sleep(ENGINE_LOOP_INTERVAL)

            if self.scheduler.empty:
                continue

            jobs = self.scheduler.schedule()
            self.runner.run_iter(jobs)
            self.scheduler.finish()
