from typing import List
import torch
from torch import nn
from xformers import ops as xops

from ..primitive_job import PrimitiveJob, Fill, Generation
from .mem import get_k_cache, get_v_cache
from .iter_state import IterationState
from .kernels import (
    discontinuous_move_tokens,
    vllm_paged_attention,
    vllm_reshape_and_cache,
)
from ..config import NativeConfig


class AttnFunc(nn.Module):
    """Base class for attention functions."""

    def __init__(
        self,
        layer_idx: int,
        scaling: float,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.scaling = scaling
        self.num_heads = num_heads
        self.head_dim = head_dim

    @staticmethod
    def init_iteration_state(
        iteration_state: IterationState,
        native_config: NativeConfig,
        jobs: List[PrimitiveJob],
        num_heads: int,
        head_size: int,
    ):
        raise NotImplementedError


class xFormersWithBuffer(AttnFunc):
    """Attention using xformers optimized operators.

    Since we manage paged KV cache, we need to first load them into a contiguous space (i.e buffer)

    NOTE: This is not a fast implementation, but it is a reference implementation for correctness.
    And it is a fusion of Fill and Generation operators.
    """

    @staticmethod
    def init_iteration_state(
        iteration_state: IterationState,
        native_config: NativeConfig,
        jobs: List[PrimitiveJob],
        num_heads: int,
        head_size: int,
    ):
        # Block Ids
        whole_ctx_block_ids: List[int] = []  # The block ids of the whole context
        newly_part_block_ids: List[int] = []  # The block ids of the newly part

        # Mask
        q_lens: List[int] = []
        kv_lens: List[int] = []

        for job in jobs:
            if isinstance(job, Fill):
                num_tokens = len(job.token_ids)
                iteration_state.num_fill_tokens.append(num_tokens)
            elif isinstance(job, Generation):
                num_tokens = 1
                iteration_state.generation_sampling_config.append(job.sampling_config)

            context_block_ids = job.context.get_context_block_ids()
            whole_ctx_block_ids.extend(context_block_ids)
            newly_part_block_ids.extend(context_block_ids[-num_tokens:])

            q_lens.append(num_tokens)
            kv_lens.append(job.context.get_context_len())

        # KV Buffer
        buffer_shape = [sum(kv_lens), num_heads, head_size]
        iteration_state.k_buffer = torch.empty(
            buffer_shape,
            dtype=native_config.dtype,
            device=native_config.device,
        )
        iteration_state.v_buffer = torch.empty(
            buffer_shape,
            dtype=native_config.dtype,
            device=native_config.device,
        )

        # Attn Mask
        iteration_state.q_kv_attn_bias = (
            xops.fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                q_seqlen=q_lens,
                kv_seqlen=kv_lens,
            )
        )

        # Indices
        iteration_state.allocated_index_tensor = torch.tensor(
            newly_part_block_ids,
            dtype=torch.int64,
            device=native_config.device,
        )
        iteration_state.context_index_tensor = torch.tensor(
            whole_ctx_block_ids,
            dtype=torch.int64,
            device=native_config.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        iteration_state: IterationState,
    ):
        k_cache = get_k_cache(self.layer_idx)
        v_cache = get_v_cache(self.layer_idx)

        # cache new k/v
        assert k.shape[0] == v.shape[0]
        src_indices = torch.arange(k.shape[0], dtype=torch.int64, device=k.device)
        discontinuous_move_tokens(
            k,
            k_cache,
            src_indices=src_indices,
            dest_indices=iteration_state.allocated_index_tensor,
        )
        discontinuous_move_tokens(
            v,
            v_cache,
            src_indices=src_indices,
            dest_indices=iteration_state.allocated_index_tensor,
        )

        # fetch cached k/v into buffer
        dest_indices = torch.arange(
            iteration_state.k_buffer.shape[0],
            dtype=torch.int64,
            device=k.device,
        )
        discontinuous_move_tokens(
            k_cache,
            iteration_state.k_buffer,
            src_indices=iteration_state.context_index_tensor,
            dest_indices=dest_indices,
        )
        discontinuous_move_tokens(
            v_cache,
            iteration_state.v_buffer,
            src_indices=iteration_state.context_index_tensor,
            dest_indices=dest_indices,
        )

        # torch.testing.assert_close(iteration_state.k_buffer[-1], k[-1])
        # NOTE(chaofan): Unsqueeze to make it compatible with xformers
        attn_output = xops.memory_efficient_attention_forward(
            q.unsqueeze(0),
            iteration_state.k_buffer.unsqueeze(0),
            iteration_state.v_buffer.unsqueeze(0),
            attn_bias=iteration_state.q_kv_attn_bias,
            p=0.0,
            scale=self.scaling,
            op=xops.fmha.cutlass.FwOp(),
        )

        return attn_output.view(-1, self.num_heads * self.head_dim)


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    return x + [pad] * (max_len - len(x))


class xFormersFill_vLLMPagedAttentionGenerate(AttnFunc):
    """Attention using xformers optimized operators and vLLM paged attention.

    This is close to the implementation of vLLM, which uses xformers operators for Fill,
    and paged attention for Generate.

    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py
    """

    @staticmethod
    def init_iteration_state(
        iteration_state: IterationState,
        native_config: NativeConfig,
        jobs: List[PrimitiveJob],
        num_heads: int,
        head_size: int,
    ):
        block_size = native_config.block_size

        # Address Tables
        block_tables = []  # [num_generation_seqs, max_num_blocks_per_seq]
        slot_mapping = []  # [num_tokens]
        context_lens = []  # [num_generation_seqs]

        # Mask
        q_lens: List[int] = []

        # Maxium
        max_num_blocks_per_seq = -1

        for job in jobs:
            if isinstance(job, Fill):
                num_tokens = len(job.token_ids)
                iteration_state.num_fill_tokens.append(num_tokens)
            elif isinstance(job, Generation):
                num_tokens = 1
                iteration_state.generation_sampling_config.append(job.sampling_config)

            context_block_ids = job.context.get_context_block_ids()
            newly_block_ids = context_block_ids[-num_tokens:]
            context_len = job.context.get_context_len()

            # Update slot mapping for query tokens
            for i, block_id in enumerate(newly_block_ids):
                offset = i + len(context_block_ids) - num_tokens
                slot_mapping.append(block_id * block_size + offset % block_size)

            if isinstance(job, Generation):
                # Update block tables for generation tokens
                # This tables is logicial block id -> physical block id, so we need to
                # squeeze the tokens to blocks
                block_tables.append(context_block_ids[::block_size])
                context_lens.append(context_len)
                max_num_blocks_per_seq = max(
                    max_num_blocks_per_seq, len(block_tables[-1])
                )
            else:
                # In this case, q_lens are only for Fill
                q_lens.append(num_tokens)

                assert (
                    context_len == num_tokens
                ), f"In vLLM, context-aware Fill is not allowed: context_len={context_len}."

        # Attn Mask
        iteration_state.q_attn_bias = (
            xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(q_lens)
        )

        # Tensors for vLLM

        # NOTE: We must pad block tables to the same length.
        block_tables = [_pad_to_max(x, max_num_blocks_per_seq, 0) for x in block_tables]

        # print(block_tables)
        # print(slot_mapping)
        # print(context_lens)

        iteration_state.block_tables = torch.tensor(
            block_tables,
            dtype=torch.int32,
            device=native_config.device,
        )

        iteration_state.slot_mapping = torch.tensor(
            slot_mapping,
            dtype=torch.int32,
            device=native_config.device,
        )

        iteration_state.context_lens = torch.tensor(
            context_lens,
            dtype=torch.int32,
            device=native_config.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        iteration_state: IterationState,
    ):
        k_cache = get_k_cache(self.layer_idx)
        v_cache = get_v_cache(self.layer_idx)

        # Cache new k/v
        vllm_reshape_and_cache(
            k,
            v,
            k_cache,
            v_cache,
            iteration_state.slot_mapping,
        )

        # Pre-allocate output
        output = torch.empty_like(q)

        # Calculate attn for Fill
        num_total_fill_tokens = iteration_state.num_total_fill_tokens
        if num_total_fill_tokens > 0:
            q_fill = q[:num_total_fill_tokens]
            fill_output = xops.memory_efficient_attention_forward(
                q_fill.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                attn_bias=iteration_state.q_attn_bias,
                p=0.0,
                scale=self.scaling,
                op=xops.fmha.cutlass.FwOp(),
            )
            output[:num_total_fill_tokens] = fill_output

        if iteration_state.num_generation_jobs > 0:
            # Calculate attn for Generate
            q_gen = q[num_total_fill_tokens:]
            head_mapping = torch.arange(
                self.num_heads, device=q_gen.device, dtype=torch.int32
            )
            gen_output = vllm_paged_attention(
                q_gen,
                k_cache,
                v_cache,
                head_mapping,
                iteration_state.context_lens,
                iteration_state.block_tables,
            )
            output[num_total_fill_tokens:] = gen_output

        return output.view(-1, self.num_heads * self.head_dim)


ATTN_FUNC_MAP = {
    "xformers_with_buffer": xFormersWithBuffer,
    "xformers_fill_vllm_paged_attention_generate": xFormersFill_vLLMPagedAttentionGenerate,
}

# NOTE(chaofan): This is a hack to make the ATTN_FUNC_MAP visible to the config.
# To avoid circular import, we cannot import ATTN_FUNC_MAP in config.py.
NativeConfig._attn_func_map = ATTN_FUNC_MAP
