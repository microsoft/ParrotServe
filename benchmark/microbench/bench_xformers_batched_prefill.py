"""In this bench we don't consider multi-heads.

Parameters: batch_size, sequence_length, hidden_dim
"""


import xformers.ops as xops
import torch
import sys
import time


def _prepare_qkv(batch_size, sequence_length, head_size, head_dim):
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(
        [batch_size, sequence_length, head_size, head_dim], dtype=dtype, device=device
    )
    k = torch.randn(
        [batch_size, sequence_length, head_size, head_dim], dtype=dtype, device=device
    )
    v = torch.randn(
        [batch_size, sequence_length, head_size, head_dim], dtype=dtype, device=device
    )

    return q, k, v


def profile_batched(q, k, v):
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    seq_lens = [seq_len for _ in range(batch_size)]
    q = q.view(1, -1, q.shape[-2], q.shape[-1])
    k = k.view(1, -1, k.shape[-2], k.shape[-1])
    v = v.view(1, -1, v.shape[-2], v.shape[-1])
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ]
    # ) as p:
    attn_bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seq_lens)
    xops.fmha.memory_efficient_attention_forward(
        q,
        k,
        v,
        attn_bias=attn_bias,
        # op=xops.fmha.cutlass.FwOp(),
    )
    # torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # r = (q @ k.mT) @ v


def profile_sequential(q, k, v):
    batch_size = q.shape[0]
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ]
    # ) as p:
    for i in range(batch_size):
        xops.fmha.memory_efficient_attention_forward(
            q[i : i + 1],
            k[i : i + 1],
            v[i : i + 1],
            attn_bias=xops.LowerTriangularMask(),
            # op=xops.fmha.cutlass.FwOp(),
        )
    # torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # r = (q @ k_mT) @ v


def main(func):
    warmups = 10
    repeats = 100

    batch_size = 20
    sequence_length = 670
    head_size = 32
    head_dim = 128

    print(
        f"batch_size: {batch_size}, sequence_length: {sequence_length}, head_size: {head_size}, head_dim: {head_dim}"
    )

    q, k, v = _prepare_qkv(batch_size, sequence_length, head_size, head_dim)

    for i in range(warmups):
        func(q, k, v)

    torch.cuda.synchronize()
    st = time.perf_counter_ns()
    for i in range(repeats):
        func(q, k, v)
    torch.cuda.synchronize()
    ed = time.perf_counter_ns()
    print(f"{func} Avg. time: {(ed - st) / repeats / 1e6} ms")


if __name__ == "__main__":
    main(profile_batched)
    main(profile_sequential)
