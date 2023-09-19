import torch
from torch import nn
from xformers import ops as xops

from ..mem import get_k_cache, get_v_cache
from ..iter_state import IterationState
from ..kernels import discontinuous_move_tokens, rotary_embedding


class xFormersWithBuffer(nn.Module):
    """Attention using xformers optimized operators.

    Since we manage paged KV cache, we need to first load them into a contiguous space (i.e buffer  )
    """

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
            attn_bias=iteration_state.x_attn_bias,
            p=0.0,
            scale=self.scaling,
            op=xops.fmha.cutlass.FwOp(),
        )

        return attn_output.view(-1, self.num_heads * self.head_dim)


class xFormersWithBufferRoPE(xFormersWithBuffer):
    """Operators with rotary position embedding."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        iteration_state: IterationState,
    ):
        assert iteration_state.cos_buffer is not None
        assert iteration_state.sin_buffer is not None
        # Should we fuse them?
        rotary_embedding(q, iteration_state.cos_buffer, iteration_state.sin_buffer)
        rotary_embedding(k, iteration_state.cos_buffer, iteration_state.sin_buffer)
        return super().forward(q, k, v, iteration_state)
