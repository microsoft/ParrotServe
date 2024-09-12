# Builtin Engine

The Builtin Engine offers several efficient implementations of popular LLMs (such as OPT and LLaMA) using PyTorch, CUDA, and Triton, including a special [customized kernel](shared_attention_kernel.md) tailored for our specific use cases.

## Server

Each Builtin Engine operates an HTTP server to handle the [internal APIs](engine_apis.md) between the Engine Layer and the Serve Layer. The engine server runs a perpetual `engine_loop`, where each iteration performs batch inference using the LLM.

## Runner

The Runner inside the engine provides a minimal interface to run the LLM via the method `run_iter(jobs: List[PrimitiveJob])`. The runner itself does not maintain any internal state; instead, it processes batches of jobs (primitive requests), and updates are recorded within the `PrimitiveJob` objects.

## Memory

### Model Weights

Parrot Builtin Engine directly loads the full model weights into GPU global memory for now. See `parrot/engine/builtin/model_instantiation.py`.

### KV Cache

Parrot Builtin Engine employs [PagedAttention](https://arxiv.org/abs/2309.06180) to divide KV Cache into blocks, storing them in the GPU’s global memory. Parrot supports different kinds of Memory layouts, such as normal layout (K and V are the same), TokenAttention (`block_size=1`), vLLM-style (The `hidden_size` dimension of the K cache is split by a constant `x` for better memory access).


### Cos/Sin Cache

For models that use RoPE (e.g., LLaMA), there is an optional cache for storing cosine and sine values. For a given `max_seq_len`, these values can be precomputed according to the RoPE algorithm and reused in all subsequent forward passes.