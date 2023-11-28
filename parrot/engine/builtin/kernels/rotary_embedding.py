# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Rotary embedding kernel implemented by Triton.

Currently only support GPT-NeoX style rotary embedding.

Why re-implement rotary embedding instead of using HF implementation?
- HF implementation is not optimized
- We want to manage cos/sin buffer

References: 
https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu
https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/rotary_emb.py
"""

import torch

import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    state,  # [num_tokens, head_num, head_dim]
    cos,  # [num_tokens, 1, head_dim // 2]
    sin,  # [num_tokens, 1, head_dim // 2]
    stride_state_n,
    stride_state_h,
    stride_state_d,
    stride_cos_n,
    stride_cos_d,
    # stride_sin_n,
    # stride_sin_d,
    num_tokens,
    num_heads,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_index = tl.program_id(0)
    token_range = token_index * BLOCK_N + tl.arange(0, BLOCK_N)
    head_index = tl.program_id(1)
    head_range = head_index * BLOCK_H + tl.arange(0, BLOCK_H)

    dim_range_x = tl.arange(0, BLOCK_D // 2)
    dim_range_y = tl.arange(BLOCK_D // 2, BLOCK_D)

    state_x_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_x[None, None, :] * stride_state_d
    )
    state_y_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_y[None, None, :] * stride_state_d
    )

    cos_sim_offset = (
        token_range[:, None, None] * stride_cos_n
        + dim_range_x[None, None, :] * stride_cos_d
    )

    state_x = tl.load(
        state + state_x_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )
    state_y = tl.load(
        state + state_y_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )

    cos_loaded = tl.load(
        cos + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    )
    sin_loaded = tl.load(
        sin + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    )

    out_x = state_x * cos_loaded - state_y * sin_loaded
    out_y = state_x * sin_loaded + state_y * cos_loaded

    tl.store(
        state + state_x_offset,
        out_x,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )
    tl.store(
        state + state_y_offset,
        out_y,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )


@torch.inference_mode()
def rotary_embedding(state, cos, sin):
    num_tokens = state.shape[0]
    num_heads = state.shape[1]
    head_dim = state.shape[2]

    BLOCK_N = 32
    BLOCK_H = 4
    grid = (
        triton.cdiv(num_tokens, BLOCK_N),
        triton.cdiv(num_heads, BLOCK_H),
    )
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    rotary_embedding_kernel[grid](
        state,
        cos,
        sin,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        cos.stride(0),
        cos.stride(2),
        # sin.stride(0),
        # sin.stride(2),
        num_tokens,
        num_heads,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def torch_rotary_embedding(state, cos, sin):
    _, _, dim = state.shape
    state_x = state[:, :, 0 : dim // 2]
    state_y = state[:, :, dim // 2 : dim]
    out_x = state_x * cos - state_y * sin
    out_y = state_x * sin + state_y * cos
    return torch.cat((out_x, out_y), dim=-1)


if __name__ == "__main__":
    tokens_num = 256
    num_heads = 96
    head_dim = 128
    max_positions = 1024

    # torch.float16 has floating point problem in Triton 2.0.0
    # But it works fine in Triton 2.1.0
    dtype = torch.float16
    state = torch.randn((tokens_num, num_heads, head_dim), dtype=dtype, device="cuda")
    cos_shape = (tokens_num, 1, head_dim // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    # forward pass
    torch_result = torch_rotary_embedding(state, cos, sin)
    rotary_embedding(state, cos, sin)
    triton_result = state  # state is modified in-place
    # print(torch_result)
    # print(triton_result)
    assert torch.allclose(torch_result, triton_result, atol=1e-2, rtol=0)
