# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Flash attention with K/V context."""


import torch

import triton
import triton.language as tl


"""A batch of Fills with dynamic sequence length is like:

|       Fill 0      |  Fill 1      |          Fill 2           |
| T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 |

T: token

Suppose:
    block_size=1,

Then: 
    num_tokens=12, (4+3+5=12)
    num_seqs=3,
    max_num_blocks_per_seq=5, 
    q_lens = [4, 3, 5]

We also need to pass the kv_lens and block_tables for loading K/V.
Suppose:
    kv_lens = [64, 128, 32]
    shape of block_tables: (3, 128) (with padding.), dtype=torch.int32
"""


@triton.jit
def _flash_attention_with_context_kernel(
    q,  # [num_tokens, num_heads, head_size]
    k_cache,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    v_cache,  # [num_blocks, num_kv_heads, head_size, block_size]
    q_lens,  # [num_seqs]
    kv_lens,  # [num_seqs]
    block_tables,  # [num_seqs, max_num_blocks_per_seq]
    out,  # [num_tokens, num_heads, head_size]
    sm_scale,
    max_num_blocks_per_seq,
    block_size,
    num_heads,
    head_size,
    x,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """TODO"""


if __name__ == "__main__":
    torch.manual_seed(2023)
