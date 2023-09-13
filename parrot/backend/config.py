from typing import Literal
from dataclasses import dataclass


@dataclass
class BackendConfig:
    cache_blocks_num: int
    attn_func: Literal["xformers_with_buffer", "flash_attention"]
    seed: int
