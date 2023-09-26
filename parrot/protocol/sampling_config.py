from dataclasses import dataclass, field
from typing import List


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = -1
    max_gen_length: int = 1024
    ignore_tokenizer_eos: bool = False
    stop_token_ids: List[int] = field(default_factory=list)
    # No support for now
    repetition_penalty: float = 0.0
    length_penalty: float = 0.0
    num_beams: int = 0
