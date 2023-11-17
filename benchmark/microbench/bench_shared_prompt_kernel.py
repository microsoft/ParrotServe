"""In this bench we don't consider multi-heads.
Parameters: batch_size, sequence_length, hidden_dim
"""


import time
import torch
import triton
import triton.language as tl
from vllm import attention_ops


@triton.jit
def _fwd_kernel_v2(
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


def triton_paged_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size / x, block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size]
    head_mapping: torch.Tensor,  # [num_heads]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    num_seqs, num_heads, head_size = query.shape
    assert head_size in {16, 32, 64, 128}
    _, num_kv_heads, _, block_size, x = key_cache.shape
    _, max_num_blocks_per_seq = block_tables.shape
    scale = head_size**-0.5
    output = torch.zeros_like(query)
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(num_seqs, BLOCK_M), num_heads)
    _fwd_kernel_v2[grid](
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


def ref_paged_attention(
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


def profile(
    batch_size: int,
    sequence_length: int,
    head_size: int,
    head_num: int = 32,
    block_size: int = 16,
    x: int = 8,
    dtype: torch.dtype = torch.float16,
    device: torch.device = "cuda",
    shared_prefix: bool = True,
    flash: bool = False,
    warmups: int = 20,
    iters: int = 100,
    seed: int = 2023,
):
    torch.manual_seed(seed)

    batch_size = int(batch_size)
    sequence_length = int(sequence_length)
    head_size = int(head_size)

    # block_num = 32768
    block_count = sequence_length // block_size

    q = torch.randn([batch_size, head_num, head_size], dtype=dtype, device=device)
    k_cache = torch.randn(
        [block_count, head_num, head_size // x, block_size, x],
        dtype=dtype,
        device=device,
    ).repeat((batch_size, 1, 1, 1, 1))
    v_cache = torch.randn(
        [block_count, head_num, head_size, block_size], dtype=dtype, device=device
    ).repeat((batch_size, 1, 1, 1))

    head_mapping = torch.arange(head_num, dtype=torch.int32, device=device)
    context_lens = torch.tensor(
        [sequence_length] * batch_size, dtype=torch.int32, device=device
    )

    if shared_prefix:
        block_tables = torch.tensor(
            list(range(block_count)) * batch_size, dtype=torch.int32, device=device
        )
    else:
        block_tables = torch.tensor(
            list(range(block_count * batch_size)), dtype=torch.int32, device=device
        )
    block_tables = block_tables.reshape(batch_size, block_count)
    # print(block_tables)

    max_context_len = block_tables.shape[-1] * block_size

    if flash:

        def run_kernel():
            return triton_paged_attention(
                q,
                k_cache,
                v_cache,
                head_mapping,
                context_lens,
                block_tables,
            )

    else:

        def run_kernel():
            output = torch.empty_like(q)
            attention_ops.single_query_cached_kv_attention(
                output,
                q,
                k_cache,
                v_cache,
                head_mapping,
                head_size**-0.5,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                None,  # alibi_slopes
            )
            return output

    ref_out = ref_paged_attention(
        q,
        k_cache,
        v_cache,
        head_mapping,
        context_lens,
        block_tables,
    )
    output = run_kernel()
    torch.testing.assert_close(ref_out, output, atol=1e-2, rtol=1e-2)

    for _ in range(warmups):
        output = run_kernel()
    torch.cuda.synchronize()
    st = time.perf_counter_ns()
    for _ in range(iters):
        output = run_kernel()
    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    return output, (ed - st) / iters / 1e3


if __name__ == "__main__":
    shape = {
        "batch_size": 128,
        "sequence_length": 512,
        "head_size": 128,
        "head_num": 32,
        "block_size": 16,
    }
    print(", ".join([f"{k}={v}" for k, v in shape.items()]))
    o_copy, latency_copy = profile(**shape, shared_prefix=False, flash=False)
    print(f"[Copy] Latency: {latency_copy:.3f} us", flush=True)
    o_shared, latency_shared = profile(**shape, shared_prefix=True, flash=False)
    print(f"[Shared] Latency: {latency_shared:.3f} us", flush=True)
    o_flash, latency_flash = profile(**shape, shared_prefix=True, flash=True)
    print(f"[Flash-Shared] Latency: {latency_flash:.3f} us", flush=True)
