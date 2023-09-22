from typing import List
import torch
from transformers import PretrainedConfig
from xformers import ops as xops

from .config import RunnerConfig
from .primitives import PrimitiveJob, Fill, Generation
from ..protocol.sampling_params import SamplingParams


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
        runner_config: RunnerConfig,
    ):
        # Metadata
        self.num_fill_tokens: List[int] = []
        self.generation_sampling_params: List[SamplingParams] = []

        # Tensors
        self.allocated_index_tensor: List[int] = []
        self.context_index_tensor: List[int] = []

        # Mask
        q_lens: List[int] = []
        kv_lens: List[int] = []

        for job in jobs:
            if isinstance(job, Fill):
                num_tokens = len(job.token_ids)
                self.num_fill_tokens.append(num_tokens)
            elif isinstance(job, Generation):
                num_tokens = 1
                self.generation_sampling_params.append(job.sampling_params)

            context_blocks = job.context.get_context_blocks()
            self.context_index_tensor.extend(context_blocks)
            self.allocated_index_tensor.extend(context_blocks[-num_tokens:])

            q_lens.append(num_tokens)
            kv_lens.append(job.context.get_context_len())

        self.allocated_index_tensor = torch.tensor(
            self.allocated_index_tensor,
            dtype=torch.int64,
            device=runner_config.device,
        )
        self.context_index_tensor = torch.tensor(
            self.context_index_tensor,
            dtype=torch.int64,
            device=runner_config.device,
        )

        num_heads = model_config.num_attention_heads
        head_size = model_config.hidden_size // num_heads

        if runner_config.attn_func == "xformers_with_buffer":
            # KV Buffer
            buffer_shape = [sum(kv_lens), num_heads, head_size]
            self.k_buffer = torch.empty(
                buffer_shape,
                dtype=runner_config.dtype,
                device=runner_config.device,
            )
            self.v_buffer = torch.empty(
                buffer_shape,
                dtype=runner_config.dtype,
                device=runner_config.device,
            )

            # Attn Mask
            self.x_attn_bias = (
                xops.fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                    q_seqlen=q_lens,
                    kv_seqlen=kv_lens,
                )
            )
        else:
            raise ValueError(
                f"Unsupported attention function {runner_config.attn_func}"
            )

        # Lazy load in RoPE arch
        self.cos_buffer = None
        self.sin_buffer = None

    @property
    def num_fill_jobs(self) -> int:
        return len(self.num_fill_tokens)

    @property
    def num_generation_jobs(self) -> int:
        return len(self.generation_sampling_params)
