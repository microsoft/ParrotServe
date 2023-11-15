# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, AsyncGenerator

from parrot.utils import get_logger
from parrot.protocol.sampling_config import SamplingConfig
from parrot.protocol.layer_apis import engine_heartbeat

from ..llm_engine import LLMEngine
from ..runtime_info import EngineRuntimeInfo
from .native_runner import NativeRunner
from .block_context import BlockContext
from ..scheduler import Scheduler
from ..primitive_job import PrimitiveJob, Fill, Generation
from ..config import NativeConfig, SchedulerConfig, EngineConfig


logger = get_logger("NativeEngine")


class NativeEngine(LLMEngine):
    """Native LLM Engine for Parrot."""

    def __init__(self, engine_config: Dict, connect_to_os: bool = True):
        super().__init__(engine_config, connect_to_os)

        native_config = NativeConfig(**engine_config.pop("instance"))
        scheduler_config = SchedulerConfig(**engine_config.pop("scheduler"))
        self.engine_config = EngineConfig(
            dtype=native_config.dtype_str,
            device=native_config.device_str,
            max_batch_size=scheduler_config.max_batch_size,
            **engine_config,
        )
        self.runner = NativeRunner(
            model_name=self.engine_config.model, config=native_config
        )
        self.scheduler = Scheduler(scheduler_config)

        self.native_config = native_config

        self._register_engine(self.engine_config)

        logger.info(
            f"NativeEngine {self.engine_config.engine_name} (id={self.engine_id}) started with config: \n"
            + "\n".join(
                [
                    f"  {key}={value}, "
                    for key, value in self.engine_config.__dict__.items()
                ]
            )
        )

    def _add_job(self, job: PrimitiveJob):
        logger.debug(f"Adding job: {job}")
        self.scheduler.add_job(job)
        self.runner.context_manager.bind_job_context(
            job,
            BlockContext,
            kv_cache_manager=self.runner.kv_cache_manager,
            block_size=self.native_config.block_size,
        )

    # ---------- Public APIs ----------

    # override
    async def fill(self, payload: Dict) -> Dict:
        fill_job = Fill(
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            token_ids=payload["token_ids"],
        )

        self._add_job(fill_job)
        await fill_job.finish_event.wait()
        return {
            "num_filled_len": len(fill_job.token_ids),
        }

    # override
    async def generate(self, payload: Dict) -> Dict:
        generation_job = Generation(
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            sampling_config=SamplingConfig(**payload["sampling_config"]),
        )
        self._add_job(generation_job)

        await generation_job.finish_event.wait()

        generated_token_ids = []
        while not generation_job.output_queue.empty():
            generated_token_ids.append(generation_job.output_queue.get())

        return {
            "generated_text": "",
            "generated_ids": generated_token_ids,
        }

    # override
    def generate_stream(self, payload: Dict) -> AsyncGenerator:
        pid = payload["pid"]
        tid = payload["tid"]
        context_id = payload["context_id"]
        parent_context_id = payload["parent_context_id"]
        sampling_config = SamplingConfig(**payload["sampling_config"])

        generation_job = Generation(
            pid=pid,
            tid=tid,
            context_id=context_id,
            parent_context_id=parent_context_id,
            sampling_config=sampling_config,
        )
        self._add_job(generation_job)

        return generation_job.generator()

    # override
    def free_context(self, payload: Dict) -> Dict:
        context_id = payload["context_id"]
        for job in self.scheduler.running_jobs:
            if job.context_id == context_id:
                # NOTE(chaofan): We cannot free the context when it is still running.
                raise RuntimeError(f"Context {context_id} is still running.")

        context_len = self.runner.context_manager.free_context(context_id)
        return {
            "context_len": context_len,
        }

    # override
    async def heartbeat(self):
        if not self.connect_to_os:
            return

        logger.debug(f"Heartbeat sent to OS (address={self.os_http_address}).")

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

        resp = await engine_heartbeat(
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

    # override
    async def engine_iter(self):
        # If there is no job, we don't need to run.
        if self.scheduler.empty:
            return

        jobs = self.scheduler.schedule()
        self.runner.run_iter(jobs)
        self.scheduler.finish()
