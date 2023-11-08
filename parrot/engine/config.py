from typing import Literal, Optional, Dict, Union
from dataclasses import dataclass
import torch

from parrot.constants import (
    FILL_NO_CHUNK,
    DEFAULT_SERVER_HOST,
    DEFAULT_ENGINE_SERVER_PORT,
)


@dataclass
class NativeConfig:
    num_kv_cache_blocks: int
    random_seed: int
    attn_func: Union[str, "AttnFunc"]
    dtype: Union[Literal["float16", "float32"], torch.dtype] = "float16"
    device: Union[str, torch.device] = "cuda"  # cpu, cuda, cuda:x
    block_size: int = 1
    attn_func_name: Optional[str] = None
    mem_layout: Optional["MemLayout"] = None
    model_arch: Optional[str] = None

    def __post_init__(self):
        # Will be overwritten by native_config_post_init.py
        pass


@dataclass
class MLCConfig:
    model_path: str
    lib_path: str
    device: str = "cuda"  # 'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto'


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

    # Forward from scheduler config
    max_batch_size: int = 1

    @classmethod
    def verify_config(cls, config: Dict) -> bool:
        """Verify the engine config."""

        if "runner" not in config or "scheduler" not in config:
            return False

        # for field in cls.__dataclass_fields__:
        #     if field in runner_keys:
        #         continue

        #     if field not in config:
        #         return False

        # Check Literal
        if config["engine_type"] not in ENGINE_TYPES:
            return False

        return True
