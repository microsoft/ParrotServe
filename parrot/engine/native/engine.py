import asyncio
import json
import time
from typing import Optional

from parrot.constants import ENGINE_LOOP_INTERVAL, ENGINE_HEARTBEAT_INTERVAL
from parrot.utils import get_logger
from parrot.protocol.layer_apis import register_engine, engine_heartbeat

from ..runtime_info import EngineRuntimeInfo
from .runner import Runner
from .block_context import BlockContext
from ..scheduler import Scheduler
from ..primitive_job import PrimitiveJob
from ..config import NativeConfig, SchedulerConfig, EngineConfig


logger = get_logger("NativeExecutionEngine")


class NativeExecutionEngine:
    """Backend Execution Engine for Parrot."""

    def __init__(self, engine_config_path: str, os_http_address: Optional[str]):
        with open(engine_config_path) as f:
            self.engine_config = dict(json.load(f))

        if not EngineConfig.verify_config(self.engine_config):
            raise ValueError(f"Invalid engine config: {self.engine_config}")

        # self.engine_name = self.engine_config["engine_name"]=
        native_config = NativeConfig(**self.engine_config.pop("runner"))
        scheduler_config = SchedulerConfig(**self.engine_config.pop("scheduler"))
        self.engine_config = EngineConfig(
            dtype=native_config.dtype_str,
            device=native_config.device_str,
            **self.engine_config,
        )
        self.runner = Runner(
            model_name=self.engine_config.model_name, config=native_config
        )
        self.scheduler = Scheduler(scheduler_config)
        self.os_http_address = os_http_address

        if self.connect_to_os:
            resp = register_engine(
                http_addr=self.os_http_address,
                engine_config=self.engine_config,
            )
            self.engine_id = resp.engine_id
        else:
            self.engine_id = 0

        logger.info(
            f"Engine {self.engine_config.engine_name} (id={self.engine_id}) started with config: \n"
            + "\n".join(
                [
                    f"  {key}={value}, "
                    for key, value in self.engine_config.__dict__.items()
                ]
            )
        )

    @property
    def connect_to_os(self):
        return self.os_http_address is not None

    def _add_job(self, job: PrimitiveJob):
        logger.info(f"Adding job: {job}")
        self.scheduler.add_job(job)
        self.runner.context_manager.bind_job_context(
            job,
            BlockContext,
            kv_cache_manager=self.runner.kv_cache_manager,
        )

    def free_context(self, context_id: int) -> int:
        for job in self.scheduler.running_jobs:
            if job.context_id == context_id:
                # NOTE(chaofan): We cannot free the context when it is still running.
                raise RuntimeError(f"Context {context_id} is still running.")

        return self.runner.context_manager.free_context(context_id)

    def heartbeat(self):
        """Return: num_cached_tokens, cached_tokens_size. num_running_jobs."""

        if not self.connect_to_os:
            return

        logger.info(f"Heartbeat sent to OS.")

        num_cached_tokens = self.runner.context_manager.get_num_cached_tokens()

        cache_mem = (
            num_cached_tokens
            # TODO(chaofan): Currently this config must be OPTConfig.
            # Support other configs in the future./
            * self.runner.hf_model_config.hidden_size
            * self.runner.hf_model_config.num_hidden_layers
            * 2
            / 1024
            / 1024
        )  # MiB
        num_running_jobs = len(self.scheduler.running_jobs)

        resp = engine_heartbeat(
            http_addr=self.os_http_address,
            engine_id=self.engine_id,
            engine_name=self.engine_config.engine_name,
            runtime_info=EngineRuntimeInfo(
                num_cached_tokens=num_cached_tokens,
                num_running_jobs=num_running_jobs,
                cache_mem=cache_mem,
                model_mem=self.runner.model_mem,  # MiB
            ),
        )

    async def engine_loop(self):
        logger.info(f"Engine loop of engine: {self.engine_config.engine_name} started.")

        last_heartbeat_time = 0  # So that we can send heartbeat at the beginning

        while True:
            # Send heartbeat to OS
            cur_time = time.perf_counter_ns()
            if (cur_time - last_heartbeat_time) / 1e9 > ENGINE_HEARTBEAT_INTERVAL:
                self.heartbeat()
                last_heartbeat_time = cur_time

            await asyncio.sleep(ENGINE_LOOP_INTERVAL)

            # Run jobs
            if self.scheduler.empty:
                continue

            jobs = self.scheduler.schedule()
            self.runner.run_iter(jobs)
            self.scheduler.finish()
