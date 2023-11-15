"""In this bench, we use a regular input to test the latency of the attention 
functions.

The input shape (query) is [num_seqs, num_heads, head_size].
The key/value cache shape is [num_blocks, num_kv_heads, head_size // block_size, block_size]. 
(For vLLM paged attention ops, the key cache is split by x.s)
"""


from parrot.engine.native.kernels import discontinuous_move_tokens
import time
import torch
from vllm import attention_ops
from xformers import ops as xops


def ref_attention_decode(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_heads, head_size]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    _, num_heads, head_size = query.shape
    scale = head_size**-0.5
    output = []
    for q, context_len, block_table in zip(query, context_lens, block_tables):
        k = key_cache[block_table]  # [max_seq_len, num_heads, head_size]
        v = value_cache[block_table]  # [max_seq_len, num_heads, head_size]
        p = torch.einsum("hd, nhd -> hn", q * scale, k).reshape((num_heads, -1))
        p[:, context_len:] = -torch.inf
        s = torch.softmax(p, dim=-1)
        o = torch.einsum("hn, nhd -> hd", s, v)
        output.append(o.unsqueeze(0))
    return torch.concat(output)


def xformers_with_buffer(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_heads, head_size]
    context_lens: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
):
    max_context_len = block_tables.shape[-1]
    batch_size, num_heads, head_size = query.shape
    scale = head_size**-0.5
    k = torch.empty(
        [batch_size * max_context_len, num_heads, head_size],
        dtype=query.dtype,
        device=query.device,
    )
    v = torch.empty(
        [batch_size * max_context_len, num_heads, head_size],
        dtype=query.dtype,
        device=query.device,
    )

    block_tables = block_tables.flatten()
    dst_indices = torch.arange(
        batch_size * max_context_len, dtype=torch.int32, device=query.device
    )

    discontinuous_move_tokens(key_cache, k, block_tables, dst_indices)
    discontinuous_move_tokens(value_cache, v, block_tables, dst_indices)

    q_lens = [1] * batch_size
    kv_lens = context_lens.tolist()

    attn_bias = xops.fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        q_seqlen=q_lens,
        kv_seqlen=kv_lens,
    )

    attn_output = xops.memory_efficient_attention_forward(
        query.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=xops.fmha.cutlass.FwOp(),
    )

    return attn_output.squeeze(0)


def profile(
    attn_func: str,
    batch_size: int,
    sequence_length: int,
    head_size: int,
    head_num: int = 32,
    block_size: int = 32,
    x: int = 8,
    dtype: torch.dtype = torch.float16,
    device: torch.device = "cuda",
    warmups: int = 20,
    iters: int = 100,
    seed: int = 2023,
):
    torch.manual_seed(seed)

    token_count = sequence_length * batch_size
    block_count = token_count // block_size

    q = torch.randn([batch_size, head_num, head_size], dtype=dtype, device=device)

    # k/v cache with normal layout
    k_cache = torch.randn(
        [token_count // batch_size, head_num, head_size], dtype=dtype, device=device
    ).repeat((batch_size, 1, 1))
    v_cache = torch.randn(
        [token_count // batch_size, head_num, head_size], dtype=dtype, device=device
    ).repeat((batch_size, 1, 1))

    # k/v cache with vllm layout
    k_cache_vllm = (
        k_cache.reshape([block_count, block_size, head_num, head_size // x, 1, x])
        .swapaxes(1, -2)
        .squeeze(1)
    )
    v_cache_vllm = (
        v_cache.reshape([block_count, block_size, head_num, head_size, 1])
        .swapaxes(1, -1)
        .squeeze(1)
    )

    print(k_cache.shape, v_cache.shape)
    print(k_cache_vllm.shape, v_cache_vllm.shape)

    head_mapping = torch.arange(head_num, dtype=torch.int32, device=device)
    context_lens = torch.tensor(
        [sequence_length] * batch_size, dtype=torch.int32, device=device
    )

    block_tables = torch.tensor(
        list(range(token_count)), dtype=torch.int32, device=device
    )

    block_tables = block_tables.reshape(batch_size, sequence_length)

    block_tables_vllm = torch.tensor(
        list(range(block_count)), dtype=torch.int32, device=device
    )
    block_tables_vllm = block_tables_vllm.reshape(
        batch_size, sequence_length // block_size
    )

    max_context_len = block_tables.shape[-1]

    def run_kernel():
        if attn_func == "vllm":
            output = torch.empty_like(q)
            attention_ops.single_query_cached_kv_attention(
                output,
                q,
                k_cache_vllm,
                v_cache_vllm,
                head_mapping,
                head_size**-0.5,
                block_tables_vllm,
                context_lens,
                block_size,  # block_size
                max_context_len,
                None,  # alibi_slopes
            )
            return output
        elif attn_func == "xformers_with_buffer":
            return xformers_with_buffer(
                q,
                k_cache,
                v_cache,
                context_lens,
                block_tables,
            )
        else:
            raise ValueError(f"Unknown attn_func: {attn_func}")

    ref_out = ref_attention_decode(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
    )
    output = run_kernel()
    # print(output)
    # print(ref_out)
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

    _, latency = profile("xformers_with_buffer", **shape)
    print(f"Latency: {latency:.3f} us")
