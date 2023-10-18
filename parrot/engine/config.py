from typing import Literal, Optional
from dataclasses import dataclass
import torch
from enum import Enum, auto

from parrot.constants import DEFAULT_SERVER_HOST, DEFAULT_ENGINE_SERVER_PORT

_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class NativeConfig:
    model_name: str
    num_kv_cache_blocks: int
    attn_func: Literal["xformers_with_buffer", "flash_attention"]
    random_seed: int
    dtype: Literal["float16", "float32"] = "float16"
    device: Literal["cuda", "cpu"] = "cuda"

    model_arch: Optional[str] = None  # Lazy load

    def __post_init__(self):
        self.dtype = _DTYPE_MAP[self.dtype]
        self.device = torch.device(self.device)


@dataclass
class SchedulerConfig:
    max_batch_size: int
    max_tokens_sum: int


class EngineType(Enum):
    NATIVE = auto()
    HUGGINGFACE = auto()
    OPENAI = auto()
    MLCLLM = auto()


@dataclass
class EngineConfig:
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_ENGINE_SERVER_PORT
    model_name: str = "unknown"
    engine_type: EngineType = EngineType.NATIVE
    tokenizer_name: str = "unknown"
