from typing import List, Dict
from dataclasses import dataclass
import torch
from transformers import OPTConfig
from xformers import ops as xops

from .mem import KVContext
from .config import AttentionConfig
from ..protocol.sampling_params import SamplingParams


@dataclass
class BackendJob:
    session_id: int
    context_id: int


@dataclass
class FillJob(BackendJob):
    parent_context_id: int
    tokens_id: List[int]


@dataclass
class GenerationJob(BackendJob):
    sampling_params: SamplingParams


@dataclass
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
        jobs: List[BackendJob],
        context_manager: Dict[int, KVContext],
        model_config: OPTConfig,
        attn_config: AttentionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # Metadata
        self.fill_tokens_num: List[int] = []
        self.generation_sampling_params: List[SamplingParams] = []

        # Tensors
        self.allocated_index_tensor: List[int] = []
        self.context_index_tensor: List[int] = []

        # Mask
        q_lens: List[int] = []
        kv_lens: List[int] = []

        for job in jobs:
            # Context
            context = context_manager[job.context_id]

            if isinstance(job, FillJob):
                tokens_num = len(job.tokens_id)
                self.fill_tokens_num.append(tokens_num)
            elif isinstance(job, GenerationJob):
                tokens_num = 1
                self.generation_sampling_params.append(job.sampling_params)

            context_blocks = context.get_context_blocks()
            self.context_index_tensor.extend(context_blocks)
            self.allocated_index_tensor.extend(context_blocks[-tokens_num:])

            q_lens.append(tokens_num)
            kv_lens.append(context.get_context_len())

        self.device = device

        self.allocated_index_tensor = torch.tensor(
            self.allocated_index_tensor, dtype=torch.int64, device=device
        )
        self.context_index_tensor = torch.tensor(
            self.context_index_tensor, dtype=torch.int64, device=device
        )

        num_heads = model_config.num_attention_heads
        head_size = model_config.hidden_size // num_heads

        if attn_config.attn_func == "xformers_with_buffer":
            # KV Buffer
            buffer_shape = [sum(kv_lens), num_heads, head_size]
            self.k_buffer = torch.empty(buffer_shape, dtype=dtype, device=device)
            self.v_buffer = torch.empty(buffer_shape, dtype=dtype, device=device)

            # Attn Mask
            self.x_attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                q_seqlen=q_lens,
                kv_seqlen=kv_lens,
            )

    @property
    def num_fill_jobs(self) -> int:
        return len(self.fill_tokens_num)

    @property
    def num_generation_jobs(self) -> int:
        return len(self.generation_sampling_params)
