# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, AsyncGenerator
import openai
import time
import asyncio

from parrot.utils import get_logger, create_task_in_loop, time_counter_in_nanoseconds
from parrot.sampling_config import SamplingConfig
from parrot.protocol.internal.runtime_info import EngineRuntimeInfo
from parrot.constants import UNKNOWN_DATA_FIELD

from .api_endpoint import Endpoint
from ..context.text_context import TextContext
from ...protocol.internal.runtime_info import EngineRuntimeInfo
from ..context.context_manager import EngineContextManager
from ..primitive_job import PrimitiveJob, Fill, Generate
from ..engine_scheduler import EngineScheduler
from ..llm_engine import LLMEngine
from ..config import OpenAIConfig, EngineConfig, SchedulerConfig
from ..latency_analyzer import LatencyAnalyzer

logger = get_logger("OpenAIEngine")


class OpenAIEngine(LLMEngine):
    """OpenAIEngine powered by OpenAI APIs."""

    def __init__(self, engine_config: Dict, connect_to_core: bool = True):
        super().__init__(engine_config, connect_to_core)

        scheduler_config = SchedulerConfig(**engine_config["scheduler"])
        scheduler_config.max_batch_size = 9999999999999  # Unlimited
        scheduler_config.max_num_batched_tokens = 9999999999999  # Unlimited
        scheduler_config.max_total_tokens = 9999999999999  # Unlimited

        # ---------- Configs ----------
        self.openai_config = OpenAIConfig(**engine_config["instance"])
        self.engine_config = EngineConfig.from_dict(engine_config)

        # ---------- Components ----------
        self.scheduler = EngineScheduler(scheduler_config)
        self.context_manager = EngineContextManager()
        # self.latency_analyzer = LatencyAnalyzer()

        # Create a OpenAI client
        logger.info(
            f"Creating an OpenAI client of the model: {self.engine_config.model} ..."
        )

        if self.openai_config.is_azure:
            self.client = openai.AsyncAzureOpenAI(
                api_key=self.openai_config.api_key,
                api_version=self.openai_config.azure_api_version,
                base_url=self.openai_config.base_url,
                azure_endpoint=self.openai_config.azure_endpoint,
            )
        else:
            self.client = openai.AsyncOpenAI(
                api_key=self.openai_config.api_key,
                base_url=self.openai_config.base_url,
            )

        self._register_engine(self.engine_config)

        logger.info(
            f"OpenAIEngine {self.engine_config.engine_name} (id={self.engine_id}) started with config: \n"
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

    async def _execute_job(self, job: PrimitiveJob):
        if isinstance(job, Fill):
            # Just fill the text context.
            job.context.append_text(job.text, role_is_user=True)
            logger.debug(f"Fill job done. Fill length: {len(job.text)}")
        elif isinstance(job, Generate):
            # Generate text and append it to the text context.

            logger.debug("Generate job started. Submit request to OpenAI API...")
            st = time_counter_in_nanoseconds()

            if self.openai_config.api_endpoint == Endpoint.COMPLETION:
                prompt = job.context.get_whole_context_text()
                logger.debug(f"Send messages: {prompt} to OpenAI API.")
                completion = await self.client.completions.create(
                    prompt=prompt,
                    model=self.engine_config.model,
                    # seed=self.engine_config.random_seed, # It is beta
                    **job.sampling_config.get_openai_params(),
                )
                generated_result = completion.choices[0].message.content
            else:
                chat_messages = job.context.get_whole_chat_messages()
                logger.debug(f"Send messages: {chat_messages} to OpenAI API.")
                chat_completion = await self.client.chat.completions.create(
                    messages=chat_messages,
                    model=self.engine_config.model,
                    # seed=self.engine_config.random_seed,
                    **job.sampling_config.get_openai_params(),
                )
                generated_result = chat_completion.choices[0].message.content

            ed = time_counter_in_nanoseconds()
            logger.debug(
                f"Generate job done. Request E2E latency: {(ed - st) / 1e9:.3f} (s)."
            )

            job.context.append_text(generated_result, role_is_user=False)
        else:
            raise NotImplementedError

        job.finish_event.set()

    # ---------- Public APIs ----------

    # override
    async def fill(self, payload: Dict) -> Dict:
        fill_job = Fill(
            session_id=payload["session_id"],
            task_id=payload["task_id"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            text=payload["text"],
        )

        self._add_job(fill_job)
        await fill_job.finish_event.wait()

        self.scheduler.finish()

        return {
            "filled_len": len(fill_job.text),
        }

    # override
    async def generate(self, payload: Dict) -> Dict:
        generation_job = Generate(
            session_id=payload["session_id"],
            task_id=payload["task_id"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            sampling_config=SamplingConfig(**payload["sampling_config"]),
        )

        self._add_job(generation_job)
        await generation_job.finish_event.wait()

        return {
            "generated_text": generation_job.context.get_latest_context_text(),
            "generated_ids": [],
        }

    # override
    async def generate_stream(self, payload: Dict) -> AsyncGenerator:
        raise NotImplementedError

    # override
    async def free_context(self, payload: Dict) -> Dict:
        context_id = payload["context_id"]
        context_len = self.context_manager.free_context(context_id)
        return {
            "context_len": context_len,
        }

    # override
    def get_runtime_info(self, profile: bool) -> EngineRuntimeInfo:
        # NOTE(chaofan): For OpenAI Engine, mem-related fields are unknown.
        num_cached_tokens = UNKNOWN_DATA_FIELD
        cache_mem = UNKNOWN_DATA_FIELD
        model_mem = UNKNOWN_DATA_FIELD

        num_running_jobs = self.scheduler.num_running_jobs
        num_total_jobs = self.scheduler.num_total_jobs

        recent_avarage_latency = 0  # self.latency_analyzer.get_average_latency()

        return EngineRuntimeInfo(
            num_cached_tokens=num_cached_tokens,
            num_running_jobs=num_running_jobs,
            num_total_jobs=num_total_jobs,
            cache_mem=cache_mem,
            model_mem=model_mem,
            recent_average_latency=recent_avarage_latency,
        )

    async def _execute_iter(self):
        jobs = self.scheduler.schedule()

        logger.debug(f"Running {len(jobs)} jobs. ")

        # coroutines = [self._execute_job(job) for job in jobs]

        for job in jobs:
            if isinstance(job, Fill):
                # Execute it immediately.
                await self._execute_job(job)
            elif isinstance(job, Generate):
                # Execute it in background.
                self.scheduler.running_jobs.remove(job)  # Avoiding repeated execution
                create_task_in_loop(self._execute_job(job))

        self.scheduler.finish()

        # if len(coroutines) > 0:
        # st = time_counter_in_nanoseconds()
        # await asyncio.gather(*coroutines)
        # ed = time_counter_in_nanoseconds()
        # iter_latency = ed - st
        # self.latency_analyzer.add_latency(iter_latency)

    # override
    async def engine_iter(self):
        """Get the jobs and execute them asynchronously."""

        if self.scheduler.is_empty:
            return

        await self._execute_iter()
