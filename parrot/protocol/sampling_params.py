# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

from dataclasses import dataclass, field
from typing import List


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_length: int = 128
    # No support for now
    repetition_penalty: float = 0.0
    length_penalty: float = 0.0
    num_beams: int = 0
    stop_token_ids: List[int] = field(default_factory=list)
