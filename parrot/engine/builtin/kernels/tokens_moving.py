# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""KV Buffer: (token_nums, head_num, head_dim).

query: (token_nums, head_num, head_dim)
key: (token_nums, head_num, head_dim)
value: (token_nums, head_num, head_dim)

References: 
https://github.com/ModelTC/lightllm/blob/main/lightllm/common/basemodel/triton_kernel/destindex_copy_kv.py
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
    src_indices,  # Shape = [num_tokens,]
    dest_indices,  # Shape = [num_tokens,]
    stride_i_n,  # Stride of src_storage along the num_tokens dimension
    stride_i_h,  # Stride of src_storage along the num_heads dimension
    stride_i_d,  # Stride of src_storage along the head_dim dimension
    stride_o_n,  # Stride of dest_storage along the num_tokens dimension
    stride_o_h,  # Stride of dest_storage along the num_heads dimension
    stride_o_d,  # Stride of dest_storage along the head_dim dimension
    num_heads,  # Number of attention heads
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Move tokens discontinuously from the input storage to the output storage."""

    token_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)

    src_index = tl.load(src_indices + token_index)  # Load src index
    dest_index = tl.load(dest_indices + token_index)  # Load dest index

    cache_ptrs = (
        src_storage
        + src_index * stride_i_n
        + stride_i_h * offs_h[:, None]
        + stride_i_d * offs_d[None, :],
    )
    out_ptrs = (
        dest_storage
        + dest_index * stride_o_n
        + stride_o_h * offs_h[:, None]
        + stride_o_d * offs_d[None, :],
    )

    token = tl.load(cache_ptrs, mask=offs_h[:, None] < num_heads, other=0.0)
    tl.store(out_ptrs, token, mask=offs_h[:, None] < num_heads)


@torch.inference_mode()
def discontinuous_move_tokens(src_storage, dest_storage, src_indices, dest_indices):
    assert (
        src_indices.shape == dest_indices.shape
    ), "src_indices and dest_indices must have the same shape"
    assert src_storage.shape[0] >= src_indices.shape[0], "src_storage is too small"
    assert dest_storage.shape[0] >= dest_indices.shape[0], "dest_storage is too small"
    assert dest_storage.shape[1:] == src_storage.shape[1:], "storage shape mismatch"

    num_tokens = src_indices.shape[0]
    num_heads, head_dim = src_storage.shape[1:]

    BLOCK_H = triton.next_power_of_2(num_heads)
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
        BLOCK_H=BLOCK_H,
        BLOCK_D=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


# Grid: num_tokens
@triton.jit
def move_tokens_from_blocked_k_cache_kernel(
    blocked_k_cache,  # Source storage. Shape = [num_blocks, head_num, head_dim // x]
    dest_storage,  # Destination storage. Shape = [num_dest_storage_tokens, head_num, head_dim // x, x]
    src_slot_indices,  # Shape = [num_tokens,]
    dest_indices,  # Shape = [num_tokens,]
    stride_kcache_n,  # Stride of k_cache along the num_blocks dimension
    stride_kcache_h,  # Stride of k_cache along the num_heads dimension
    stride_kcache_d,  # Stride of k_cache along the head_dim // x dimension
    stride_kcache_b,  # Stride of v_cache along the block_size dimension
    stride_kcache_x,  # Stride of dest_storage along the x dimension
    stride_o_n,  # Stride of dest_storage along the num_tokens dimension
    stride_o_h,  # Stride of dest_storage along the num_heads dimension
    stride_o_d,  # Stride of dest_storage along the head_dim dimension
    stride_o_x,  # Stride of dest_storage along the x dimension
    block_size,  # Block size
    num_heads,  # Number of attention heads
    BLOCK_H: tl.constexpr,
    BLOCK_D_DIV_X: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    """Move tokens discontinuously from the input storage to the output storage."""

    token_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    x_idx = tl.arange(0, BLOCK_D_DIV_X)
    x_offs = tl.arange(0, BLOCK_X)

    src_index = tl.load(src_slot_indices + token_index)  # Load src index
    dest_index = tl.load(dest_indices + token_index)  # Load dest index

    block_id = src_index // block_size
    block_offset = src_index % block_size

    cache_ptrs = (
        blocked_k_cache
        + block_id * stride_kcache_n
        + stride_kcache_h * offs_h[:, None, None]
        + stride_kcache_d * x_idx[None, :, None]
        + stride_kcache_b * block_offset
        + stride_kcache_x * x_offs[None, None, :]
    )

    out_ptrs = (
        dest_storage
        + dest_index * stride_o_n
        + stride_o_h * offs_h[:, None, None]
        + stride_o_d * x_idx[None, :, None]
        + stride_o_x * x_offs[None, None, :],
    )
    token = tl.load(cache_ptrs, mask=offs_h[:, None, None] < num_heads, other=0.0)
    # token = tl.view(token, [BLOCK_H, BLOCK_D])
    # tl.device_print("Token: ", token)
    # tl.static_print("Token: ", token)
    # tl.static_print("Cache ptrs: ", cache_ptrs)
    # tl.static_print("Out ptrs: ", out_ptrs)
    tl.store(out_ptrs, token, mask=offs_h[:, None, None] < num_heads)


@torch.no_grad()
def move_tokens_from_blocked_k_cache(
    blocked_k_cache, dest_storage, src_slot_indices, dest_indices
):
    assert (
        src_slot_indices.shape == dest_indices.shape
    ), "src_indices and dest_indices must have the same shape"

    num_heads, head_dim_div_x, block_size, x = blocked_k_cache.shape[1:]

    # assert (
    #     blocked_k_cache.shape[0] * block_size >= src_slot_indices.shape[0]
    # ), "blocked_k_cache is too small"

    # Reshape dest_storage into vLLM layout
    original_shape = dest_storage.shape
    dest_storage = dest_storage.view(
        dest_storage.shape[0], dest_storage.shape[1], dest_storage.shape[2] // x, x
    )

    assert dest_storage.shape[0] >= dest_indices.shape[0], "dest_storage is too small"
    assert dest_storage.shape[1] == blocked_k_cache.shape[1], "storage shape mismatch"

    num_tokens = src_slot_indices.shape[0]

    BLOCK_H = triton.next_power_of_2(num_heads)
    grid = (num_tokens,)
    num_warps = 1

    move_tokens_from_blocked_k_cache_kernel[grid](
        blocked_k_cache,
        dest_storage,
        src_slot_indices,
        dest_indices,
        blocked_k_cache.stride(0),
        blocked_k_cache.stride(1),
        blocked_k_cache.stride(2),
        blocked_k_cache.stride(3),
        blocked_k_cache.stride(4),
        dest_storage.stride(0),
        dest_storage.stride(1),
        dest_storage.stride(2),
        dest_storage.stride(3),
        block_size,
        num_heads,
        BLOCK_H=BLOCK_H,
        BLOCK_D_DIV_X=head_dim_div_x,
        BLOCK_X=x,
        num_warps=num_warps,
        num_stages=1,
    )

    # Reshape dest_storage back to the original layout
    dest_storage = dest_storage.view(original_shape)

    return


# Grid: num_tokens
@triton.jit
def move_tokens_from_blocked_v_cache_kernel(
    blocked_v_cache,  # Source storage. Shape = [num_blocks, head_num, head_dim]
    dest_storage,  # Destination storage. Shape = [num_dest_storage_tokens, head_num, head_dim]
    src_slot_indices,  # Shape = [num_tokens,]
    dest_indices,  # Shape = [num_tokens,]
    stride_vcache_n,  # Stride of v_cache along the num_blocks dimension
    stride_vcache_h,  # Stride of v_cache along the num_heads dimension
    stride_vcache_d,  # Stride of v_cache along the head_dim dimension
    stride_vcache_b,  # Stride of v_cache along the block_size dimension
    stride_o_n,  # Stride of dest_storage along the num_tokens dimension
    stride_o_h,  # Stride of dest_storage along the num_heads dimension
    stride_o_d,  # Stride of dest_storage along the head_dim dimension
    block_size,  # Block size
    num_heads,  # Number of attention heads
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Move tokens discontinuously from the input storage to the output storage."""

    token_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)

    src_index = tl.load(src_slot_indices + token_index)  # Load src index
    dest_index = tl.load(dest_indices + token_index)  # Load dest index

    block_id = src_index // block_size
    block_offset = src_index % block_size

    cache_ptrs = (
        blocked_v_cache
        + block_id * stride_vcache_n
        + stride_vcache_h * offs_h[:, None]
        + stride_vcache_d * offs_d[None, :]
        + stride_vcache_b * block_offset
    )

    out_ptrs = (
        dest_storage
        + dest_index * stride_o_n
        + stride_o_h * offs_h[:, None]
        + stride_o_d * offs_d[None, :],
    )

    tokens = tl.load(cache_ptrs, mask=offs_h[:, None] < num_heads, other=0.0)
    tl.store(out_ptrs, tokens, mask=offs_h[:, None] < num_heads)


@torch.no_grad()
def move_tokens_from_blocked_v_cache(
    blocked_v_cache, dest_storage, src_slot_indices, dest_indices
):
    assert (
        src_slot_indices.shape == dest_indices.shape
    ), "src_indices and dest_indices must have the same shape"

    num_tokens = src_slot_indices.shape[0]
    num_heads, head_dim, block_size = blocked_v_cache.shape[1:]

    # assert (
    #     blocked_v_cache.shape[0] * block_size >= src_slot_indices.shape[0]
    # ), "blocked_v_cache is too small"
    assert dest_storage.shape[0] >= dest_indices.shape[0], "dest_storage is too small"
    assert (
        dest_storage.shape[1:] == blocked_v_cache.shape[1:-1]
    ), "storage shape mismatch"

    BLOCK_H = triton.next_power_of_2(num_heads)
    grid = (num_tokens,)
    num_warps = 1

    move_tokens_from_blocked_v_cache_kernel[grid](
        blocked_v_cache,
        dest_storage,
        src_slot_indices,
        dest_indices,
        blocked_v_cache.stride(0),
        blocked_v_cache.stride(1),
        blocked_v_cache.stride(2),
        blocked_v_cache.stride(3),
        dest_storage.stride(0),
        dest_storage.stride(1),
        dest_storage.stride(2),
        block_size,
        num_heads,
        BLOCK_H=BLOCK_H,
        BLOCK_D=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def test_discontinuous_move_tokens():
    torch.manual_seed(2023)

    # OPT-175B
    num_heads = 96
    head_dim = 128

    kv_cache_num_tokens = 131072 * 10  # 30 GB
    src_storage = torch.randn(
        [kv_cache_num_tokens, num_heads, head_dim], dtype=torch.float16, device="cuda"
    )
    batch_tokens = 131072  # tokens in one iteration
    dest_storage = torch.zeros(
        [batch_tokens * 2, num_heads, head_dim], dtype=torch.float16, device="cuda"
    )

    src_indices = torch.randint(
        0,
        kv_cache_num_tokens - 1,
        [batch_tokens],
        dtype=torch.int64,
        device="cuda",
    )  # Random src index

    dest_indices = torch.arange(
        0, batch_tokens * 2, 2, dtype=torch.int64, device="cuda"
    )  # Sequential dest index

    # print("Src index", src_indices)
    # print("Dest index", dest_indices)

    # discontinuous_move_tokens(src_storage, dest_storage, src_indices, dest_indices)
    # print(dest_storage) / 0

    for i in range(10):
        discontinuous_move_tokens(src_storage, dest_storage, src_indices, dest_indices)

    torch.cuda.synchronize()
    st = time.perf_counter_ns()
    for i in range(100):
        discontinuous_move_tokens(src_storage, dest_storage, src_indices, dest_indices)
    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(
        f"Move {batch_tokens * num_heads * head_dim * 2 / 1024 / 1024 / 1024} GB tokens. Time {(ed - st) / 100 / 1e9:.3f} s"
    )

    # print(dest_storage)


def test_move_tokens_from_blocked_k_cache():
    torch.manual_seed(2023)

    # OPT-175B
    num_heads = 96
    head_dim = 128

    kv_cache_num_blocks = 8192
    block_size = 16
    x = 8

    blocked_k_cache = torch.randn(
        [kv_cache_num_blocks, num_heads, head_dim // x, block_size, x],
        dtype=torch.float16,
        device="cuda",
    )
    batch_tokens = 131072  # tokens in one iteration
    dest_storage = torch.zeros(
        [batch_tokens * 2, num_heads, head_dim], dtype=torch.float16, device="cuda"
    )

    src_slot_indices = torch.randint(
        0,
        kv_cache_num_blocks * block_size - 1,
        [batch_tokens],
        dtype=torch.int64,
        device="cuda",
    )  # Random src index

    dest_indices = torch.arange(
        0, batch_tokens * 2, 2, dtype=torch.int64, device="cuda"
    )  # Sequential dest index

    # print("Src index", src_indices)
    # print("Dest index", dest_indices)

    # move_tokens_from_blocked_k_cache(
    #     blocked_k_cache, dest_storage, src_slot_indices, dest_indices
    # )
    # print(dest_storage) / 0

    for i in range(10):
        move_tokens_from_blocked_k_cache(
            blocked_k_cache, dest_storage, src_slot_indices, dest_indices
        )

    torch.cuda.synchronize()
    st = time.perf_counter_ns()
    for i in range(100):
        move_tokens_from_blocked_k_cache(
            blocked_k_cache, dest_storage, src_slot_indices, dest_indices
        )
    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(
        f"Move {batch_tokens * num_heads * head_dim * 2 / 1024 / 1024 / 1024} GB tokens. Time {(ed - st) / 100 / 1e9:.3f} s"
    )

    # print(dest_storage)


def test_move_tokens_from_blocked_v_cache():
    torch.manual_seed(2023)

    # OPT-175B
    num_heads = 96
    head_dim = 128

    kv_cache_num_blocks = 8192
    block_size = 16
    blocked_v_cache = torch.randn(
        [kv_cache_num_blocks, num_heads, head_dim, block_size],
        dtype=torch.float16,
        device="cuda",
    )
    batch_tokens = 131072  # tokens in one iteration
    dest_storage = torch.zeros(
        [batch_tokens * 2, num_heads, head_dim], dtype=torch.float16, device="cuda"
    )

    src_slot_indices = torch.randint(
        0,
        kv_cache_num_blocks * block_size - 1,
        [batch_tokens],
        dtype=torch.int64,
        device="cuda",
    )  # Random src index

    dest_indices = torch.arange(
        0, batch_tokens * 2, 2, dtype=torch.int64, device="cuda"
    )  # Sequential dest index

    # print("Src index", src_indices)
    # print("Dest index", dest_indices)

    # move_tokens_from_blocked_v_cache(
    #     blocked_v_cache, dest_storage, src_slot_indices, dest_indices
    # )
    # print(dest_storage) / 0

    for i in range(10):
        move_tokens_from_blocked_v_cache(
            blocked_v_cache, dest_storage, src_slot_indices, dest_indices
        )

    torch.cuda.synchronize()
    st = time.perf_counter_ns()
    for i in range(100):
        move_tokens_from_blocked_v_cache(
            blocked_v_cache, dest_storage, src_slot_indices, dest_indices
        )
    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(
        f"Move {batch_tokens * num_heads * head_dim * 2 / 1024 / 1024 / 1024} GB tokens. Time {(ed - st) / 100 / 1e9:.3f} s"
    )

    # print(dest_storage)


if __name__ == "__main__":
    test_discontinuous_move_tokens()
    test_move_tokens_from_blocked_k_cache()
    test_move_tokens_from_blocked_v_cache()
