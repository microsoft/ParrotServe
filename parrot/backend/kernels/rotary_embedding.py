"""Rotary embedding kernel implemented by Triton.

Currently only support GPT-NeoX style rotary embedding.

References: 
https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu
https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/rotary_emb.py
"""

import torch

import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    query,  # [num_tokens, head_num, head_dim]
    positions,  # [num_tokens,]
    cos_cache,  # [max_postions, 1, head_dim // 2]
    sin_cache,  # [max_postions, 1, head_dim // 2]
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_pos_n,
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

    q_x_offset = (
        token_range[:, None, None] * stride_q_n
        + head_range[None, :, None] * stride_q_h
        + dim_range_x[None, None, :] * stride_q_d
    )
    q_y_offset = (
        token_range[:, None, None] * stride_q_n
        + head_range[None, :, None] * stride_q_h
        + dim_range_y[None, None, :] * stride_q_d
    )

    pos = tl.load(
        positions + token_range * stride_pos_n,
        mask=token_range < num_tokens,
        other=0.0,
    )
    cos_sim_offset = (
        pos[:, None, None] * stride_cos_n + dim_range_x[None, None, :] * stride_cos_d
    )
    # cos_sim_offset = (
    #     token_range[:, None, None] * stride_cos_n
    #     + dim_range_x[None, None, :] * stride_cos_d
    # )

    q_x = tl.load(
        query + q_x_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )
    q_y = tl.load(
        query + q_y_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )

    cos = tl.load(
        cos_cache + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    )
    sin = tl.load(
        sin_cache + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    )

    out_x = q_x * cos - q_y * sin
    out_y = q_x * sin + q_y * cos

    tl.store(
        query + q_x_offset,
        out_x,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )
    tl.store(
        query + q_y_offset,
        out_y,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )


@torch.no_grad()
def rotary_embedding(query, positions, cos_cache, sin_cache):
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]

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
        query,
        positions,
        cos_cache,
        sin_cache,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        positions.stride(0),
        cos_cache.stride(0),
        cos_cache.stride(2),
        # sin_cache.stride(0),
        # sin_cache.stride(2),
        num_tokens,
        num_heads,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def torch_rotary_embedding(query, positions, cos_cache, sin_cache):
    _, _, dim = query.shape
    q_x = query[:, :, 0 : dim // 2]
    q_y = query[:, :, dim // 2 : dim]
    cos = cos_cache[positions]  # .view(-1, 1, dim // 2)
    sin = sin_cache[positions]  # .view(-1, 1, dim // 2)
    out_x = q_x * cos - q_y * sin
    out_y = q_x * sin + q_y * cos
    return torch.cat((out_x, out_y), dim=-1)


if __name__ == "__main__":
    tokens_num = 256
    num_heads = 96
    head_dim = 128
    max_positions = 1024

    dtype = torch.float32
    query = torch.randn((tokens_num, num_heads, head_dim), dtype=dtype, device="cuda")
    # positions = torch.arange(tokens_num, dtype=torch.int64, device="cuda")
    positions = torch.randint(
        0, max_positions, (tokens_num,), dtype=torch.int64, device="cuda"
    )
    cos_shape = (max_positions, 1, head_dim // 2)
    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    # forward pass
    torch_result = torch_rotary_embedding(query, positions, cos, sin)
    rotary_embedding(query, positions, cos, sin)
    triton_result = query  # query is modified in-place
    # print(torch_result)
    # print(triton_result)
    assert torch.allclose(torch_result, triton_result, atol=1e-2, rtol=0)
