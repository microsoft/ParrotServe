import asyncio
from typing import Dict, List
import json
import time

from parrot.constants import ENGINE_LOOP_INTERVAL, GC_INTERVAL
from parrot.utils import get_logger

from .runner import Runner
from .block_context import BlockContext
from ..scheduler import Scheduler
from ..primitive_job import PrimitiveJob
from ..low_level_context import get_unique_context_id
from ..config import NativeConfig, SchedulerConfig


logger = get_logger("NativeExecutionEngine")


class NativeExecutionEngine:
    """Backend Execution Engine for Parrot."""

    def __init__(self, engine_config_path: str):
        with open(engine_config_path) as f:
            engine_config = json.load(f)

        self.engine_name = engine_config["engine_name"]
        native_config = NativeConfig(**engine_config["runner"])
        scheduler_config = SchedulerConfig(**engine_config["scheduler"])
        self.runner = Runner(native_config)
        self.scheduler = Scheduler(scheduler_config)

    def add_job(self, job: PrimitiveJob):
        logger.info(f"Adding job: {job}")
        self.scheduler.add_job(job)
        self.runner.context_manager.bind_job_context(
            job,
            BlockContext,
            kv_cache_manager=self.runner.kv_cache_manager,
        )

    def free_context(self, client_id: str, context_id: int) -> int:
        for job in self.scheduler.running_jobs:
            if job.client_id == client_id and job.context_id == context_id:
                # NOTE(chaofan): We cannot free the context when it is still running.
                raise RuntimeError(
                    f"Context {get_unique_context_id(client_id, context_id)} is still running."
                )

        return self.runner.context_manager.free_context(client_id, context_id)

    def heartbeat(self, engine_name: str, client_id: str) -> Dict[str, int]:
        """Return: num_cached_tokens, cached_tokens_size. num_running_jobs."""

        logger.info(f"Heartbeat from client {client_id}.")

        if engine_name != self.engine_name:
            logger.warning(
                f"Name mismatch! Heart with engine name {engine_name}, "
                f"but this engine is {self.engine_name}."
            )

        self.runner.context_manager.flush_context_heartbeat(client_id)

        # NOTE(chaofan): This info is for all tokens in this engine,
        # not only for this client.
        num_cached_tokens = self.runner.context_manager.get_num_cached_tokens()

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

        last_gc_time = 0

        while True:
            await asyncio.sleep(ENGINE_LOOP_INTERVAL)

            # Do GC
            cur_time = time.perf_counter_ns()
            if cur_time - last_gc_time > GC_INTERVAL * 1e9:
                self.runner.context_manager.garbage_collect()
                last_gc_time = cur_time

            if self.scheduler.empty:
                continue

            # Run jobs
            jobs = self.scheduler.schedule()
            self.runner.run_iter(jobs)
            self.scheduler.finish()
