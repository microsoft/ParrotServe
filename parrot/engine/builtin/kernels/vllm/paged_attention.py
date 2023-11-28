# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import torch
from vllm import attention_ops, cache_ops


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


# def profile_attention(
#     fn, mask_nnz, mode="fwd", warmup=25, rep=100, num_heads=48, head_size=64
# ):
#     ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
#     flops_per_matmul = 2.0 * num_heads * mask_nnz * head_size
#     total_flops = 2 * flops_per_matmul
#     if mode == "bwd":
#         total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
#     gflops = total_flops / ms * 1e-9
#     print(f"{mode}: {ms:.3f} ms | {gflops:.3f} GFLOP/s")


def test_attention(dtype=torch.float16, device="cuda", kernel=vllm_paged_attention):
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

    torch.testing.assert_close(o, ref_o, atol=1e-2, rtol=1e-2)

    # print(ref_o)
    # print(o)

    # import ipdb; ipdb.set_trace()
    # mask_nnz = context_lens.sum().item()
    # profile_attention(
    #     forward_fn, mask_nnz, "fwd", num_heads=num_heads, head_size=head_size
    # )


if __name__ == "__main__":
    torch.manual_seed(2023)

    test_attention(kernel=vllm_paged_attention)
