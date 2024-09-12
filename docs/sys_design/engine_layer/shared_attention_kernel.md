# Shared Attention Kernel

SharedAttention is a kernel optimization algorithm that we proposed in our paper to optimize the case that a batch of requests sharing the same system prompt. There are some great concurrent works which also propose similar ideas like us: [Cascade Inference](https://docs.flashinfer.ai/tutorials/recursive_attention.html), [Chunk Attention](https://arxiv.org/abs/2402.15220), which readers can check to better understand this algorithm.

## Implementation Sketch

In Parrot's SharedAttention, we calculate the Attention in two steps. For example, if we have a batch of requests A, B, C which share the same system prompt S:
- Step 1: Calculate the partial Attention of `S` (batch size = 1) using Flash Attention. There are some intermediate results like `qk_max` and `exp_sum` we need to store.
- Step 2: Calculate the batched partial Attention of the diverged part of A, B, C (batch size = 3) using PagedAttention. The merging process of the these two partial Attentions is also finished in this kernel so we need to pass `qk_max` and `exp_sum` to this kernel as input.

Note 1: There are two orders of this algorithm: first Flash then Paged, or first Paged then Flash. Our experiment shows the impact of these two orders on performance is negligible.

Note 2: Actually we can parallelize the Flash part and the Paged part, followed by a separate step (which could be called "Merge") to complete the operation.