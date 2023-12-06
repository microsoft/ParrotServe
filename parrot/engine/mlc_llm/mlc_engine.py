# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import time
from typing import Dict, AsyncGenerator


MLC_INSTALLED = True
try:
    from mlc_chat import ChatModule, GenerationConfig
    from mlc_chat.chat_module import logging

    logging.info = (
        lambda x: x
    )  # This is a dirty way to disable logging in MLC-LLM, avoiding repeative logging.
except ImportError:
    MLC_INSTALLED = False

import torch
import psutil

from parrot.utils import get_logger
from parrot.protocol.sampling_config import SamplingConfig
from parrot.protocol.runtime_info import EngineRuntimeInfo
from parrot.constants import NONE_CONTEXT_ID, UNKNOWN_DATA_FIELD
from parrot.exceptions import ParrotEngineInternalError

from ..context.text_context import TextContext
from ..context.context_manager import ContextManager
from ..latency_analyzer import LatencyAnalyzer
from ..scheduler import Scheduler
from ..llm_engine import LLMEngine
from ..primitive_job import PrimitiveJob, Fill, Generate
from ..config import MLCConfig, EngineConfig, SchedulerConfig


logger = get_logger("MLCEngine")


def get_memory(is_gpu: bool = True, gpu_no: int = 0) -> int:
    """Returns the total memory of the GPU/CPU in bytes."""
    if is_gpu:
        return torch.cuda.get_device_properties(gpu_no).total_memory
    else:
        return psutil.virtual_memory().total


class MLCEngine(LLMEngine):
    """MLC LLM Engine for Parrot powered by MLC-LLM.

    Reference (MLC-LLM): https://llm.mlc.ai
    """

    def __init__(self, engine_config: Dict, connect_to_os: bool = True):
        if not MLC_INSTALLED:
            raise ParrotEngineInternalError(
                "MLC is not installed hence can not start a MLC-LLM engine."
            )

        super().__init__(engine_config, connect_to_os)

        scheduler_config = engine_config.pop("scheduler")
        scheduler_config = SchedulerConfig(**scheduler_config)
        scheduler_config.max_batch_size = 999999  # Batching for MLC-LLM is meaningless
        scheduler_config.max_num_batched_tokens = 9999999999999  # Unlimited

        # ---------- Configs ----------
        mlc_config = MLCConfig(**engine_config.pop("instance"))
        self.engine_config = EngineConfig(
            dtype="float16",  # A fake dtype for now
            device=mlc_config.device,
            **engine_config,
        )

        # Create a ChatModule instance
        logger.info(
            f"Creating a ChatModule instance of the model: {self.engine_config.model} ..."
        )

        # TODO: Using config info
        is_gpu = False
        gpu_no = 0

        # start_memory = get_memory(is_gpu, gpu_no)
        self.chat_module = ChatModule(
            model=mlc_config.model_path,
            model_lib_path=mlc_config.lib_path,
            device=mlc_config.device,
        )
        # end_memory = get_memory(is_gpu, gpu_no)
        self.model_mem = UNKNOWN_DATA_FIELD

        # logger.info(
        #     f"Model {self.engine_config.model_name} loaded. Total size: {self.model_mem} MiB"
        # )

        self._current_context_id = NONE_CONTEXT_ID

        # ---------- Components ----------
        self.scheduler = Scheduler(scheduler_config)
        self.latency_analyzer = LatencyAnalyzer()
        self.context_manager = ContextManager()

        self._register_engine(self.engine_config)

        logger.info(
            f"MLCEngine {self.engine_config.engine_name} (id={self.engine_id}) started with config: \n"
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
        self.context_manager.bind_job_context(
            job,
            TextContext,
        )

    def _execute_job(self, job: PrimitiveJob):
        if isinstance(job, Fill):
            st = time.perf_counter_ns()

            self.chat_module._prefill(job.text)
            job.context.append_text(job.text, role_is_user=True)

            ed = time.perf_counter_ns()
            logger.debug(
                f"Fill time: {(ed - st) / 1e9:.3f} (s). Stats: {self.chat_module.stats()}"
            )
        elif isinstance(job, Generate):
            generation_config = GenerationConfig(
                temperature=job.sampling_config.temperature,
                top_p=job.sampling_config.top_p,
                max_gen_len=job.sampling_config.max_gen_length,
            )
            while not self.chat_module._stopped():
                st = time.perf_counter_ns()
                self.chat_module._decode(generation_config=generation_config)
                ed = time.perf_counter_ns()
                logger.debug(
                    f"Generate time: {(ed - st) / 1e9:.3f} (s). Stats: {self.chat_module.stats()}"
                )
            generated_text = self.chat_module._get_message()
            job.context.append_text(generated_text, role_is_user=False)
        else:
            raise NotImplementedError

        job.finish_event.set()

    # ---------- Public APIs ----------

    async def fill(self, payload: Dict) -> Dict:
        context_id = payload["context_id"]

        if self._current_context_id == NONE_CONTEXT_ID:
            self._current_context_id = context_id
        else:
            assert (
                self._current_context_id == context_id
            ), "MLCEngine only supports one active context."

        fill_job = Fill(
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=context_id,
            parent_context_id=payload["parent_context_id"],
            text=payload["text"],
        )

        # self._execute_job(fill_job)
        self._add_job(fill_job)

        await fill_job.finish_event.wait()

        return {
            "filled_len": len(fill_job.text),
        }

    # override
    async def generate(self, payload: Dict) -> Dict:
        context_id = payload["context_id"]

        assert self._current_context_id != NONE_CONTEXT_ID, "No active context."
        assert (
            self._current_context_id == context_id
        ), "MLCEngine only supports one active context."

        generation_job = Generate(
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=context_id,
            parent_context_id=payload["parent_context_id"],
            sampling_config=SamplingConfig(**payload["sampling_config"]),
        )

        # self._execute_job(generation_job)
        self._add_job(generation_job)

        await generation_job.finish_event.wait()

        generated_text = generation_job.context.get_latest_context_text()

        return {
            "generated_text": generated_text,
            "generated_ids": [],
        }

    # override
    def generate_stream(self, payload: Dict) -> AsyncGenerator:
        # MLC-LLM doesn't support token-level streaming generation.
        raise NotImplementedError

    # override
    async def free_context(self, payload: Dict) -> Dict:
        context_id = payload["context_id"]

        assert self._current_context_id != NONE_CONTEXT_ID, "No active context."
        assert (
            self._current_context_id == context_id
        ), "MLCEngine only supports one active context."

        self._current_context_id = NONE_CONTEXT_ID
        self.context_manager.free_context(context_id)

        self.chat_module.reset_chat()
        return {
            "context_len": 0,
        }

    def get_runtime_info(self, profile: bool) -> EngineRuntimeInfo:
        num_cached_tokens = UNKNOWN_DATA_FIELD
        num_running_jobs = self.scheduler.num_running_jobs
        num_total_jobs = self.scheduler.num_total_jobs
        cache_mem = UNKNOWN_DATA_FIELD
        recent_average_latency = self.latency_analyzer.get_average_latency()

        return EngineRuntimeInfo(
            num_cached_tokens=num_cached_tokens,
            num_running_jobs=num_running_jobs,
            num_total_jobs=num_total_jobs,
            cache_mem=cache_mem,
            model_mem=self.model_mem,
            recent_average_latency=recent_average_latency,
        )

    async def engine_iter(self):
        # If there is no job, we don't need to run.
        if self.scheduler.empty:
            return

        jobs = self.scheduler.schedule()
        selected_job = None

        for job in jobs:
            if (
                job.context_id == self._current_context_id
                or self._current_context_id == NONE_CONTEXT_ID
            ):
                selected_job = job
                break

        st = time.perf_counter_ns()
        self._execute_job(selected_job)
        ed = time.perf_counter_ns()
        e2e_time = ed - st

        self.latency_analyzer.add_latency(e2e_time)
        self.scheduler.finish()
