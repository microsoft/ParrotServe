"""KV Buffer: (token_nums, head_num, head_dim).

query: (token_nums, head_num, head_dim)
key: (token_nums, head_num, head_dim)
value: (token_nums, head_num, head_dim)

Reference: https://github.com/ModelTC/lightllm/blob/main/lightllm/common/basemodel/triton_kernel/destindex_copy_kv.py
"""

import torch
import time

import triton
import triton.language as tl


# Grid: num_tokens
@triton.jit
def discontinuous_move_tokens_kernel(
    src_storage,  # Source storage. Shape = [num_src_storage_tokens, head_num, head_dim]
    dest_storage,  # Destination storage. Shape = [num_dest_storage_tokens, head_num, head_dim]
    src_indices,  # Shape = [num_tokens, head_num, head_dim]
    dest_indices,  # Shape = [num_tokens, head_num, head_dim]
    stride_i_n,  # Stride of src_storage along the num_tokens dimension
    stride_i_h,  # Stride of src_storage along the num_heads dimension
    stride_i_d,  # Stride of src_storage along the d_model dimension
    stride_o_n,  # Stride of dest_storage along the num_tokens dimension
    stride_o_h,  # Stride of dest_storage along the num_heads dimension
    stride_o_d,  # Stride of dest_storage along the d_model dimension
    num_heads,  # Number of attention heads
    BLOCK_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Move tokens discontinuously from the input storage to the output storage."""

    token_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    src_indices = tl.load(src_indices + token_index)  # Load src index
    dest_indices = tl.load(dest_indices + token_index)  # Load dest index

    cache_ptrs = (
        src_storage
        + src_indices * stride_i_n
        + stride_i_h * offs_h[:, None]
        + stride_i_d * offs_d[None, :],
    )
    out_ptrs = (
        dest_storage
        + dest_indices * stride_o_n
        + stride_o_h * offs_h[:, None]
        + stride_o_d * offs_d[None, :],
    )

    tokens = tl.load(cache_ptrs, mask=offs_h[:, None] < num_heads, other=0.0)
    tl.store(out_ptrs, tokens, mask=offs_h[:, None] < num_heads)


@torch.no_grad()
def discontinuous_move_tokens(src_storage, dest_storage, src_indices, dest_indices):
    assert (
        src_indices.shape == dest_indices.shape
    ), "src_indices and dest_indices must have the same shape"
    assert src_storage.shape[0] >= src_indices.shape[0], "src_storage is too small"
    assert dest_storage.shape[0] >= dest_indices.shape[0], "dest_storage is too small"
    assert dest_storage.shape[1:] == src_storage.shape[1:], "storage shape mismatch"

    num_tokens = src_indices.shape[0]
    num_heads, d_model = src_storage.shape[1:]

    BLOCK_HEAD = triton.next_power_of_2(num_heads)
    grid = (num_tokens,)
    num_warps = 1

    discontinuous_move_tokens_kernel[grid](
        src_storage,
        dest_storage,
        src_indices,
        dest_indices,
        src_storage.stride(0),
        src_storage.stride(1),
        src_storage.stride(2),
        dest_storage.stride(0),
        dest_storage.stride(1),
        dest_storage.stride(2),
        num_heads,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_DMODEL=d_model,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test_discontinuous_move_tokens():
    # OPT-175B
    num_heads = 96
    d_model = 128

    kv_cache_tokens_num = 131072 * 10  # 30 GB
    src_storage = torch.ones(
        [kv_cache_tokens_num, num_heads, d_model], dtype=torch.float16, device="cuda"
    )
    batch_tokens = 131072  # tokens in one iteration
    dest_storage = torch.zeros(
        [batch_tokens * 2, num_heads, d_model], dtype=torch.float16, device="cuda"
    )

    src_indices = torch.randint(
        0,
        kv_cache_tokens_num - 1,
        [batch_tokens],
        dtype=torch.int64,
        device="cuda",
    )  # Random src index

    dest_indices = torch.arange(
        0, batch_tokens * 2, 2, dtype=torch.int64, device="cuda"
    )  # Sequential dest index

    # print("Src index", src_indices)
    # print("Dest index", dest_indices)

    for i in range(10):
        discontinuous_move_tokens(src_storage, dest_storage, src_indices, dest_indices)

    st = time.perf_counter_ns()
    for i in range(100):
        discontinuous_move_tokens(src_storage, dest_storage, src_indices, dest_indices)
    ed = time.perf_counter_ns()

    print(
        f"Move {batch_tokens * num_heads * d_model * 2 / 1024 / 1024 / 1024} GB tokens. Time {(ed - st) / 100 / 1e9} s"
    )

    # print(dest_storage)


if __name__ == "__main__":
    test_discontinuous_move_tokens()
