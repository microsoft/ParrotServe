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
def _fwd_kernel(
    Q,  # [num_seqs, num_heads, head_size]
    K,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    V,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping,  # [num_heads]
    context_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    Out,  # [num_seqs, num_heads, head_size]
    sm_scale,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    x,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_group_id = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    start_m = seq_group_id * BLOCK_M

    kv_head_id = tl.load(head_mapping + head_id)
    context_len = tl.load(context_lens + start_m + offs_m)  # [BLOCK_M]

    offs_q = (
        (start_m + offs_m[:, None]) * num_heads * head_size
        + head_id * head_size
        + offs_d[None, :]
    )  # [BLOCK_M, BLOCK_DMODEL]
    offs_k = (
        kv_head_id * head_size * block_size
        + (offs_d[None, :] // x) * block_size * x
        + (offs_n[:, None] % block_size) * x
        + (offs_d[None, :] % x)
    )  # [BLOCK_N, BLOCK_DMODEL]
    offs_v = (
        kv_head_id * head_size * block_size
        + offs_d[:, None] * block_size
        + (offs_n[None, :] % block_size)
    )  # [BLOCK_DMODEL, BLOCK_N]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    acc = tl.load(Out + offs_q).to(tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q + offs_q)  # [BLOCK_M, BLOCK_DMODEL]
    q = (q * qk_scale).to(tl.float16)

    for start_n in range(0, tl.max(context_len), BLOCK_N):
        # -- load block table --
        physical_block_idx = tl.load(
            block_tables + (start_n + offs_n) // block_size
        )  # [BLOCK_N]
        offs_page = (
            physical_block_idx * num_kv_heads * head_size * block_size
        )  # [BLOCK_N]
        # -- load k, v --
        k = tl.load(K + offs_k + offs_page[:, None])  # [BLOCK_N, BLOCK_DMODEL]
        v = tl.load(V + offs_v + offs_page[None, :])  # [BLOCK_DMODEL, BLOCK_N]
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(
            start_n + offs_n[None, :] < context_len[:, None], qk, float("-inf")
        )
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

    acc /= l_i[:, None]
    tl.store(Out + offs_q, acc.to(tl.float16))


# FIXME: Wrong in the unshared case


def paged_flash_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    num_seqs, num_heads, head_size = query.shape
    assert head_size in {16, 32, 64, 128}, "Unsupported head size"
    _, num_kv_heads, _, block_size, x = key_cache.shape
    _, max_num_blocks_per_seq = block_tables.shape
    scale = head_size**-0.5
    output = torch.zeros_like(query)
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(num_seqs, BLOCK_M), num_heads)
    _fwd_kernel[grid](
        query,
        key_cache,
        value_cache,
        head_mapping,
        context_lens,
        block_tables,
        output,
        scale,
        max_num_blocks_per_seq,
        block_size,
        num_heads,
        num_kv_heads,
        head_size,
        x,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=head_size,
        num_warps=4,
        num_stages=4,
    )
    return output


### Paged Flash Attention End ###


### vLLM Paged Attention Begin ###


def vllm_paged_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    output = torch.empty_like(query)
    _, _, head_size = query.shape
    assert head_size in {16, 32, 64, 128}, "Unsupported head size"
    _, _, _, block_size, _ = key_cache.shape
    max_context_len = block_tables.shape[-1] * block_size
    attention_ops.single_query_cached_kv_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        head_size**-0.5,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # alibi_slopes
    )
    return output


def vllm_reshape_and_cache(
    key_to_cache: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value_to_cache: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    slot_mapping: torch.Tensor,  # [num_tokens]
):
    cache_ops.reshape_and_cache(
        key_to_cache,
        value_to_cache,
        key_cache,
        value_cache,
        slot_mapping,
    )


### vLLM Paged Attention End ###


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


def test_attention(dtype=torch.float16, device="cuda", kernel=paged_flash_attention):
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


def test_reshape_and_cache(dtype=torch.float16, device="cuda"):
    num_tokens = 1024
    num_blocks = 4096
    num_heads = 48
    head_size = 64
    x = 8
    block_size = 16
    key_to_cache = torch.randn(
        (num_tokens, num_heads, head_size), dtype=dtype, device=device
    )
    value_to_cache = torch.randn(
        (num_tokens, num_heads, head_size), dtype=dtype, device=device
    )
    key_cache = torch.randn(
        (num_blocks, num_heads, head_size // x, block_size, x),
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn(
        (num_blocks, num_heads, head_size, block_size), dtype=dtype, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)
    cache_ops.reshape_and_cache(
        key_to_cache,
        value_to_cache,
        key_cache,
        value_cache,
        slot_mapping,
    )
    test_token = 47
    value_ref = value_to_cache[test_token]
    slot = slot_mapping[test_token]
    block_num = slot // block_size
    value_cached = value_cache[block_num, :, :, slot % block_size]
    torch.testing.assert_close(value_ref, value_cached, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    torch.manual_seed(2023)

    test_attention(kernel=vllm_paged_attention)
    # test_attention(kernel=paged_flash_attention)
    test_reshape_and_cache()
