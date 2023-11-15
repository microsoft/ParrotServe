# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, AsyncGenerator
import openai
import time
import asyncio

from parrot.utils import get_logger
from parrot.protocol.layer_apis import engine_heartbeat
from parrot.protocol.sampling_config import SamplingConfig

from .api_endpoint import Endpoint
from .text_context import TextContext
from ..runtime_info import EngineRuntimeInfo
from ..context_manager import ContextManager
from ..primitive_job import PrimitiveJob, Fill, Generation
from ..scheduler import Scheduler
from ..llm_engine import LLMEngine
from ..config import OpenAIConfig, EngineConfig, SchedulerConfig


logger = get_logger("OpenAIEngine")


class OpenAIEngine(LLMEngine):
    """OpenAIEngine powered by OpenAI APIs."""

    def __init__(self, engine_config: Dict, connect_to_os: bool = True):
        super().__init__(engine_config, connect_to_os)

        scheduler_config = engine_config.pop("scheduler")
        scheduler_config["max_tokens_sum"] = 9999999999999  # Unlimited
        scheduler_config = SchedulerConfig(**scheduler_config)

        self.openai_config = OpenAIConfig(**engine_config.pop("instance"))
        self.engine_config = EngineConfig(
            dtype="unknown",
            device="unknown",
            **engine_config,
        )
        self.scheduler = Scheduler(scheduler_config)
        self.context_manager = ContextManager()

        # Create a OpenAI client
        logger.info(
            f"Creating an OpenAI client of the model: {self.engine_config.model} ..."
        )

        if self.openai_config.is_azure:
            self.client = openai.AsyncAzureOpenAI(
                api_key=self.openai_config.api_key,
                api_version=self.openai_config.azure_api_version,
                azure_endpoint=self.openai_config.azure_endpoint,
            )
        else:
            self.client = openai.AsyncOpenAI(api_key=self.openai_config.api_key)

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
        elif isinstance(job, Generation):
            # Generate text and append it to the text context.

            logger.debug("Generate job started. Submit request to OpenAI API...")
            st = time.perf_counter_ns()

            if self.openai_config.api_endpoint == Endpoint.COMPLETION:
                prompt = job.context.get_whole_context_text()
                completion = await self.client.completions.create(
                    prompt=prompt,
                    model=self.engine_config.model,
                    # seed=self.engine_config.random_seed, # It is beta
                    **job.sampling_config.get_openai_params(),
                )
                generated_result = completion.choices[0].message.content
            else:
                chat_messages = job.context.get_chat_messages()
                chat_completion = await self.client.chat.completions.create(
                    messages=chat_messages,
                    model=self.engine_config.model,
                    # seed=self.engine_config.random_seed,
                    **job.sampling_config.get_openai_params(),
                )
                generated_result = chat_completion.choices[0].message.content

            ed = time.perf_counter_ns()
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
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            text=payload["text"],
        )

        self._add_job(fill_job)
        await fill_job.finish_event.wait()
        return {
            "num_filled_len": len(fill_job.text),
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

        return {
            "generated_text": generation_job.context.get_latest_context_text(),
            "generated_ids": [],
        }

    # override
    def generate_stream(self, payload: Dict) -> AsyncGenerator:
        raise NotImplementedError

    # override
    def free_context(self, payload: Dict) -> Dict:
        context_id = payload["context_id"]
        context_len = self.context_manager.free_context(context_id)
        return {
            "context_len": context_len,
        }

    # override
    async def heartbeat(self):
        if not self.connect_to_os:
            return

        logger.debug(f"Heartbeat sent to OS (address={self.os_http_address}).")

        # NOTE(chaofan): For OpenAI Engine, mem-related fields are all fake.
        num_cached_tokens = 0
        cache_mem = 0  # MiB

        num_running_jobs = len(self.scheduler.running_jobs)

        resp = await engine_heartbeat(
            http_addr=self.os_http_address,
            engine_id=self.engine_id,
            engine_name=self.engine_config.engine_name,
            runtime_info=EngineRuntimeInfo(
                num_cached_tokens=num_cached_tokens,
                num_running_jobs=num_running_jobs,
                cache_mem=cache_mem,
                model_mem=0,  # MiB
            ),
        )

    # override
    async def engine_iter(self):
        """Get the jobs and execute them asynchronously."""
        # If there is no job, we don't need to run.
        if self.scheduler.empty:
            return

        jobs = self.scheduler.schedule()

        logger.debug(f"Running {len(jobs)} jobs. ")

        coroutines = [self._execute_job(job) for job in jobs]
        if len(coroutines) > 0:
            await asyncio.gather(*coroutines)

        self.scheduler.finish()
