# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List
import torch
from transformers import PretrainedConfig

from parrot.protocol.sampling_config import SamplingConfig

from ..config import BuiltinConfig
from ..primitive_job import PrimitiveJob, Fill, Generate


class IterationState:
    """Structure of an iteration:

    | ---- fill tokens ----- | ---- generation tokens ---- |
    |    F1   |   F2  |  F3  | G1 | G2 | G3 | G4 | G5 | G6 |

    F: fill tokens
    G: generation tokens

    Each fill (F1, F2, ...) is a list of tokens.
    Each generation (G1, G2, ...) is a single token.
    Every backend job has a context.
    """

    def __init__(
        self,
        jobs: List[PrimitiveJob],
        model_config: PretrainedConfig,
        builtin_config: BuiltinConfig,
    ):
        # Metadata
        self.num_fill_tokens: List[int] = []
        self.generation_sampling_config: List[SamplingConfig] = []

        num_heads = model_config.num_attention_heads
        head_size = model_config.hidden_size // num_heads

        builtin_config.attn_func.init_iteration_state(
            self,
            builtin_config,
            jobs,
            num_heads,
            head_size,
        )

    @property
    def num_fill_jobs(self) -> int:
        return len(self.num_fill_tokens)

    @property
    def num_generation_jobs(self) -> int:
        return len(self.generation_sampling_config)

    @property
    def num_total_fill_tokens(self) -> int:
        return sum(self.num_fill_tokens)
