# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

09/13/2023: support variable sequence lengths for prefix (chengzhang@microsoft.com)

TODO(chaofan): debug and integrate into Parrot.
"""


import torch

import triton
import triton.language as tl
from vllm import attention_ops, cache_ops


### Paged Flash Attention Begin ###


@triton.jit
def _fwd_kernel_v2(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    V,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping,  # [num_heads]
    context_len,
    qk_max,  # [num_seqs, num_heads]
    exp_sum,  # [num_seqs, num_heads]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, head_size]
    sm_scale,
    max_num_blocks_per_seq,
    block_size,
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    x,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOAD_MID_RESULTS: tl.constexpr,
):
    seq_group_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    start_m = seq_group_id * BLOCK_M
    m_mask = start_m + offs_m < num_seqs

    kv_head_id = tl.load(head_mapping + head_id)

    offs_q = (start_m + offs_m[:, None]) * num_heads * head_size + \
        head_id * head_size + offs_d[None, :]  # [BLOCK_M, BLOCK_DMODEL]
    offs_k = kv_head_id * head_size * block_size + (offs_d[None, :] // x) * block_size * x + \
        (offs_n[:, None] % block_size) * x + (offs_d[None, :] % x)  # [BLOCK_N, BLOCK_DMODEL]
    offs_v = kv_head_id * head_size * block_size + offs_d[:, None] * block_size + \
        (offs_n[None, :] % block_size)  # [BLOCK_DMODEL, BLOCK_N]

    if LOAD_MID_RESULTS:
        m_i = tl.load(qk_max + (start_m + offs_m) * num_heads + head_id, mask=m_mask)
        l_i = tl.load(exp_sum + (start_m + offs_m) * num_heads + head_id, mask=m_mask)
        acc = tl.load(Out + offs_q, mask=m_mask[:, None]).to(tl.float32) * l_i[:, None]
    else:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q + offs_q, mask=m_mask[:, None])  # [BLOCK_M, BLOCK_DMODEL]
    q = (q * qk_scale).to(tl.float16)

    for start_n in range(0, context_len, BLOCK_N):
        # -- load block table --
        physical_block_idx = tl.load(block_tables + (start_n + offs_n) // block_size)  # [BLOCK_N]
        offs_page = physical_block_idx * num_kv_heads * head_size * block_size  # [BLOCK_N]
        # -- load k, v --
        k = tl.load(K + offs_k + offs_page[:, None])  # [BLOCK_N, BLOCK_DMODEL]
        v = tl.load(V + offs_v + offs_page[None, :])  # [BLOCK_DMODEL, BLOCK_N]
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(start_n + offs_n[None, :] < context_len, qk, float("-inf"))
        qk += tl.dot(q, k.T)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])  # [BLOCK_M, BLOCK_N]
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v.T)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    if not LOAD_MID_RESULTS:
        tl.store(qk_max + (start_m + offs_m) * num_heads + head_id, m_i, mask=m_mask)
        tl.store(exp_sum + (start_m + offs_m) * num_heads + head_id, l_i, mask=m_mask)

    acc /= l_i[:, None]
    tl.store(Out + offs_q, acc.to(tl.float16), mask=m_mask[:, None])


def triton_flash_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_len: int,
    qk_max: torch.Tensor,  # [num_seqs, num_heads]
    exp_sum: torch.Tensor,  # [num_seqs, num_heads]
    block_table: torch.Tensor,  # [max_num_blocks_per_seq]
    output: torch.Tensor,  # [num_seqs, num_heads, head_size]
    load_mid_results: bool,
):
    num_seqs, num_heads, head_size = query.shape
    assert head_size in {16, 32, 64, 128}
    _, num_kv_heads, _, block_size, x = key_cache.shape
    max_num_blocks_per_seq = block_table.shape[0]
    scale = head_size ** -0.5
    if num_seqs <= 32:
        BLOCK_M = 32
        BLOCK_N = 32
        NUM_STAGES = 6
    elif num_seqs <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
        NUM_STAGES = 4
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        NUM_STAGES = 4
    NUM_WARPS = 4
    grid = (triton.cdiv(num_seqs, BLOCK_M), num_heads)
    _fwd_kernel_v2[grid](
        query, key_cache, value_cache, head_mapping, context_len, qk_max, exp_sum, block_table, output,
        scale, max_num_blocks_per_seq, block_size, num_seqs, num_heads, num_kv_heads, head_size, x,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_size, LOAD_MID_RESULTS=load_mid_results,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )


def flash_paged_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    flash_context_len: int,
    flash_block_table: torch.Tensor,  # [max_num_blocks_per_seq]
    paged_context_lens: torch.Tensor,  # [num_seqs]
    paged_block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    num_seqs, num_heads, head_size, block_size = value_cache.shape
    max_context_len = paged_block_tables.shape[-1] * block_size
    qk_max = torch.zeros([num_seqs, num_heads], dtype=torch.float32, device=query.device)
    exp_sum = torch.zeros([num_seqs, num_heads], dtype=torch.float32, device=query.device)
    output = torch.empty_like(query)
    triton_flash_attention(
        query,
        key_cache,
        value_cache,
        head_mapping,
        flash_context_len,
        qk_max,
        exp_sum,
        flash_block_table,
        output,
        load_mid_results=False,
    )
    attention_ops.single_query_cached_kv_post_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        head_size**-0.5,
        paged_block_tables,
        paged_context_lens,
        qk_max,
        exp_sum,
        block_size,
        max_context_len,
        None,  # alibi_slopes
    )
    return output


def paged_flash_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    flash_context_len: int,
    flash_block_table: torch.Tensor,  # [max_num_blocks_per_seq]
    paged_context_lens: torch.Tensor,  # [num_seqs]
    paged_block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    num_seqs, num_heads, head_size, block_size = value_cache.shape
    max_context_len = paged_block_tables.shape[-1] * block_size
    qk_max = torch.zeros([num_seqs, num_heads], dtype=torch.float32, device=query.device)
    exp_sum = torch.zeros([num_seqs, num_heads], dtype=torch.float32, device=query.device)
    output = torch.empty_like(query)
    attention_ops.single_query_cached_kv_prev_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        head_size**-0.5,
        paged_block_tables,
        paged_context_lens,
        qk_max,
        exp_sum,
        block_size,
        max_context_len,
        None,  # alibi_slopes
    )
    triton_flash_attention(
        query,
        key_cache,
        value_cache,
        head_mapping,
        flash_context_len,
        qk_max,
        exp_sum,
        flash_block_table,
        output,
        load_mid_results=True,
    )
    return output


### Paged Flash Attention End ###


def paged_flash_attention_reference(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    _, num_heads, head_size = query.shape
    scale = head_size**-0.5
    output = []
    for q, context_len, block_table in zip(query, context_lens, block_tables):
        v = value_cache[block_table]
        k = key_cache[block_table].swapaxes(-1, -2).reshape(v.shape)
        v = v[:, head_mapping].swapaxes(1, -1).reshape(-1, head_size, num_heads)
        k = k[:, head_mapping].swapaxes(1, -1).reshape(-1, head_size, num_heads)

        p = torch.einsum("hd, ndh -> hn", q * scale, k).reshape((num_heads, -1))
        p[:, context_len:] = -torch.inf
        s = torch.softmax(p, dim=-1)
        o = torch.einsum("hn, ndh -> hd", s, v)
        output.append(o.unsqueeze(0))
    return torch.concat(output)


def profile_attention(
    fn, mask_nnz, mode="fwd", warmup=25, rep=100, num_heads=48, head_size=64
):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * num_heads * mask_nnz * head_size
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    gflops = total_flops / ms * 1e-9
    print(f"{mode}: {ms:.3f} ms | {gflops:.3f} GFLOP/s")


def test_attention(dtype=torch.float16, device="cuda", kernel=None):
    num_seqs = 16
    num_blocks = 4096
    num_heads = 48
    head_size = 64
    x = 8
    block_size = 16
    seq_len = 1024

    q = torch.randn(
        (num_seqs, num_heads, head_size), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (num_blocks, num_heads, head_size // x, block_size, x),
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        (num_blocks, num_heads, head_size, block_size), dtype=dtype, device=device
    )

    # print(k.shape)
    # print(v.shape)

    head_mapping = torch.arange(num_heads, dtype=torch.int32, device=device)
    context_lens = torch.tensor([seq_len] * num_seqs, dtype=torch.int32, device=device)
    block_tables = torch.tensor(
        list(range(seq_len // block_size * num_seqs)), dtype=torch.int32, device=device
    ).reshape(num_seqs, -1)

    # print(block_tables)

    ref_o = paged_flash_attention_reference(
        q, k, v, head_mapping, context_lens, block_tables
    )
    forward_fn = lambda: kernel(q, k, v, head_mapping, context_lens, block_tables)
    o = forward_fn()

    # print(ref_o)
    # print(o)

    # import ipdb; ipdb.set_trace()
    torch.testing.assert_close(o, ref_o, atol=1e-2, rtol=1e-2)
    mask_nnz = context_lens.sum().item()
    profile_attention(
        forward_fn, mask_nnz, "fwd", num_heads=num_heads, head_size=head_size
    )


if __name__ == "__main__":
    torch.manual_seed(2023)

    # test_attention(kernel=vllm_paged_attention)
    # test_attention(kernel=paged_flash_attention)
    # test_reshape_and_cache()
