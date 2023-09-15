from typing import Literal
from dataclasses import dataclass


@dataclass
class RunnerConfig:
    model_name: str
    num_kv_cache_blocks: int
    attn_func: Literal["xformers_with_buffer", "flash_attention"]
    random_seed: int


@dataclass
class SchedulerConfig:
    max_batch_size: int
    max_tokens_sum: int
