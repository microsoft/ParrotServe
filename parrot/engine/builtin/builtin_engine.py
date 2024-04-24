# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, AsyncGenerator

from parrot.utils import get_logger, MemTracker, get_cpu_memory_usage, cprofile
from parrot.protocol.sampling_config import SamplingConfig
from parrot.protocol.runtime_info import EngineRuntimeInfo
from parrot.constants import UNKNOWN_DATA_FIELD

from ..llm_engine import LLMEngine
from .builtin_runner import BuiltinRunner
from ..latency_analyzer import LatencyAnalyzer
from ..context.block_context import BlockContext
from ..scheduler import Scheduler
from ..primitive_job import PrimitiveJob, Fill, Generate
from ..config import BuiltinConfig, SchedulerConfig, EngineConfig


logger = get_logger("BuiltinEngine")


class BuiltinEngine(LLMEngine):
    """Parrot built-in LLM Engine, supporting the most fine-grained level optimization."""

    def __init__(self, engine_config: Dict, connect_to_os: bool = True):
        super().__init__(engine_config, connect_to_os)

        # ---------- Configs ----------
        builtin_config = BuiltinConfig(**engine_config.pop("instance"))
        scheduler_config = SchedulerConfig(**engine_config.pop("scheduler"))
        self.engine_config = EngineConfig(
            dtype=builtin_config.dtype_str,
            device=builtin_config.device_str,
            **engine_config,
        )
        self.builtin_config = builtin_config

        # ---------- Components ----------
        self.runner = BuiltinRunner(
            model_name=self.engine_config.model, config=builtin_config
        )
        self.scheduler = Scheduler(scheduler_config)
        self.latency_analyzer = LatencyAnalyzer()
        self.gpu_mem_tracker = MemTracker(device=self.runner.local_rank)

        self._register_engine(self.engine_config)

        logger.info(
            f"BuiltinEngine {self.engine_config.engine_name} (id={self.engine_id}) started with config: \n"
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
            block_size=self.builtin_config.block_size,
        )

    # ---------- Public APIs ----------

    # override
    async def fill(self, payload: Dict) -> Dict:
        fill_job = Fill(
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            end_flag=payload["end_flag"],
            token_ids=payload["token_ids"],
        )

        self._add_job(fill_job)
        await fill_job.finish_event.wait()
        return {
            "filled_len": len(fill_job.token_ids),
        }

    # override
    async def generate(self, payload: Dict) -> Dict:
        generation_job = Generate(
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            sampling_config=SamplingConfig(**payload["sampling_config"]),
            end_flag=payload["end_flag"],
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
        end_flag = payload["end_flag"]

        generation_job = Generate(
            pid=pid,
            tid=tid,
            context_id=context_id,
            parent_context_id=parent_context_id,
            sampling_config=sampling_config,
            end_flag=end_flag,
        )
        self._add_job(generation_job)

        return generation_job.generator()

    # override
    async def free_context(self, payload: Dict) -> Dict:
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
    def get_runtime_info(self, profile: bool) -> EngineRuntimeInfo:
        # Scheduler
        num_running_jobs = self.scheduler.num_running_jobs
        num_total_jobs = self.scheduler.num_total_jobs

        # Memory
        num_cached_tokens = self.runner.context_manager.get_num_cached_tokens()
        num_max_blocks = self.runner.kv_cache_manager.get_history_max_allocated_num()
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
        model_mem = self.runner.model_mem

        recent_average_latency = self.latency_analyzer.get_average_latency()

        if profile:
            self.gpu_mem_tracker.clear_cache()
            profiled_cpu_mem = get_cpu_memory_usage()
            profiled_gpu_allocate_mem = self.gpu_mem_tracker.get_allocate_usage()
            profiled_gpu_tensor_mem = self.gpu_mem_tracker.get_tensor_usage()
        else:
            profiled_cpu_mem = UNKNOWN_DATA_FIELD
            profiled_gpu_allocate_mem = UNKNOWN_DATA_FIELD
            profiled_gpu_tensor_mem = UNKNOWN_DATA_FIELD

        return EngineRuntimeInfo(
            num_cached_tokens=num_cached_tokens,
            num_max_blocks=num_max_blocks,
            num_running_jobs=num_running_jobs,
            num_total_jobs=num_total_jobs,
            cache_mem=cache_mem,
            model_mem=model_mem,
            profiled_cpu_mem=profiled_cpu_mem,
            profiled_gpu_allocate_mem=profiled_gpu_allocate_mem,
            profiled_gpu_tensor_mem=profiled_gpu_tensor_mem,
            recent_average_latency=recent_average_latency,
        )

    # override
    async def engine_iter(self):
        # If there is no job, we don't need to run.
        if self.scheduler.empty:
            return

        jobs = self.scheduler.schedule()

        # with cprofile("run_iter"):
        e2e_time, model_time = self.runner.run_iter(jobs)

        self.latency_analyzer.add_latency(e2e_time)
        self.scheduler.finish()
