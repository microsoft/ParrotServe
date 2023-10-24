from typing import Dict, AsyncGenerator
from mlc_chat import ChatModule, GenerationConfig
from mlc_chat.chat_module import logging

logging.info = (
    lambda x: x
)  # This is a dirty way to disable logging in MLC-LLM, avoiding repeative logging.

import torch
import psutil

from parrot.utils import get_logger
from parrot.protocol.sampling_config import SamplingConfig
from parrot.protocol.layer_apis import engine_heartbeat

from ..llm_engine import LLMEngine
from ..runtime_info import EngineRuntimeInfo
from ..primitive_job import PrimitiveJob, Fill, Generation
from ..config import MLCConfig, EngineConfig


logger = get_logger("MLCEngine")


def get_memory(is_gpu: bool = True, gpu_no: int = 0) -> int:
    """Returns the total memory of the GPU/CPU in bytes."""
    if is_gpu:
        return torch.cuda.get_device_properties(gpu_no).total_memory
    else:
        return psutil.virtual_memory().total


class MLCEngine(LLMEngine):
    """MLC LLM Engine for Parrot."""

    def __init__(self, engine_config: Dict, connect_to_os: bool = True):
        super().__init__(engine_config, connect_to_os)

        scheduler_config = engine_config.pop("scheduler")

        mlc_config = MLCConfig(**engine_config.pop("runner"))
        self.engine_config = EngineConfig(
            dtype="float16",  # A fake dtype for now
            device=mlc_config.device,
            **engine_config,
        )

        # Create a ChatModule instance
        logger.info(
            f"Creating a ChatModule instance of the model: {self.engine_config.model_name} ..."
        )

        # TODO: Using config info
        is_gpu = False
        gpu_no = 0

        start_memory = get_memory(is_gpu, gpu_no)
        self.chat_module = ChatModule(
            model=mlc_config.model_path,
            model_lib_path=mlc_config.lib_path,
            device=mlc_config.device,
        )
        end_memory = get_memory(is_gpu, gpu_no)
        self.model_mem = (end_memory - start_memory) / 1024 / 1024  # MiB

        # logger.info(
        #     f"Model {self.engine_config.model_name} loaded. Total size: {self.model_mem} MiB"
        # )

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

    def _execute_job(self, job: PrimitiveJob):
        if isinstance(job, Fill):
            self.chat_module._prefill(job.text)
        elif isinstance(job, Generation):
            generation_config = GenerationConfig(
                temperature=job.sampling_config.temperature,
                top_p=job.sampling_config.top_p,
                max_gen_len=job.sampling_config.max_gen_length,
            )
            while not self.chat_module._stopped():
                self.chat_module._decode(generation_config=generation_config)
        else:
            raise NotImplementedError

    # ---------- Public APIs ----------

    async def fill(self, payload: Dict) -> Dict:
        fill_job = Fill(
            pid=payload["pid"],
            tid=payload["tid"],
            context_id=payload["context_id"],
            parent_context_id=payload["parent_context_id"],
            text=payload["text"],
        )

        self._execute_job(fill_job)

        return {
            "num_filled_tokens": len(fill_job.text),
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

        self._execute_job(generation_job)

        generated_text = self.chat_module._get_message()

        return {
            "generated_text": generated_text,
            "generated_ids": [],
        }

    # override
    def generate_stream(self, payload: Dict) -> AsyncGenerator:
        # MLC-LLM doesn't support token-level streaming generation.
        raise NotImplementedError

    # override
    def free_context(self, payload: Dict) -> Dict:
        context_id = payload["context_id"]
        self.chat_module.reset_chat()
        return {
            "num_freed_tokens": 0,
        }

    def heartbeat(self):
        """Return: num_cached_tokens, cached_tokens_size. num_running_jobs."""

        if not self.connect_to_os:
            return

        logger.info(f"Heartbeat sent to OS.")

        # TODO(chaofan): Profile real data
        num_cached_tokens = 0
        cache_mem = 0.0
        num_running_jobs = 0

        resp = engine_heartbeat(
            http_addr=self.os_http_address,
            engine_id=self.engine_id,
            engine_name=self.engine_config.engine_name,
            runtime_info=EngineRuntimeInfo(
                num_cached_tokens=num_cached_tokens,
                num_running_jobs=num_running_jobs,
                cache_mem=cache_mem,
                model_mem=self.model_mem,  # MiB
            ),
        )

    def engine_iter(self):
        # Do nothing
        pass
