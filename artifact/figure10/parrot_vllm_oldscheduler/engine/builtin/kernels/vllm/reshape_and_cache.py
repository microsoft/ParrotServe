# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import torch
from vllm import cache_ops


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

    test_reshape_and_cache()
