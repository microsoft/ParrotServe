from typing import Literal, Optional, Dict
from dataclasses import dataclass
import torch
from enum import Enum
from parrot.constants import FILL_NO_CHUNK

from parrot.constants import DEFAULT_SERVER_HOST, DEFAULT_ENGINE_SERVER_PORT

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class NativeConfig:
    num_kv_cache_blocks: int
    attn_func: Literal["xformers_with_buffer", "flash_attention"]
    random_seed: int
    dtype: Literal["float16", "float32"] = "float16"
    device: str = "cuda"  # cpu, cuda, cuda:x
    model_arch: Optional[str] = None  # Lazy load

    def __post_init__(self):
        self.dtype_str = self.dtype
        self.device_str = self.device
        self.dtype = _DTYPE_MAP[self.dtype]
        self.device = torch.device(self.device)


@dataclass
class SchedulerConfig:
    max_batch_size: int
    max_tokens_sum: int


# EngineType(Enum)
ENGINE_TYPE_NATIVE = "native"
ENGINE_TYPE_HUGGINGFACE = "huggingface"
ENGINE_TYPE_OPENAI = "openai"
ENGINE_TYPE_MLCLLM = "mlcllm"
ENGINE_TYPES = [
    ENGINE_TYPE_NATIVE,
    ENGINE_TYPE_HUGGINGFACE,
    ENGINE_TYPE_OPENAI,
    ENGINE_TYPE_MLCLLM,
]


@dataclass
class EngineConfig:
    model_name: str = "unknown"
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_ENGINE_SERVER_PORT
    engine_name: str = "unknown"
    engine_type: str = ENGINE_TYPE_NATIVE
    tokenizer_name: str = "unknown"
    fill_chunk_size: int = FILL_NO_CHUNK

    # Forward from runner config
    dtype: Literal["float16", "float32"] = "float16"
    device: str = "cuda"  # cpu, cuda, cuda:x

    @classmethod
    def verify_config(cls, config: Dict) -> bool:
        """Verify the engine config."""

        runner_keys = ["dtype", "device"]

        if "runner" not in config or "scheduler" not in config:
            return False

        for field in cls.__dataclass_fields__:
            if field in runner_keys:
                continue

            if field not in config:
                return False

        # Check Literal
        if config["engine_type"] not in ENGINE_TYPES:
            return False

        return True
