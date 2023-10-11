from typing import Literal, Optional
from dataclasses import dataclass
import torch

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


@dataclass
class EngineConfig:
    model_name: str
    engine_type: Literal["native", "hf", "openai", "mlcllm"]
