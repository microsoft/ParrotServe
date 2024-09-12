# Builtin Engine

The Builtin Engine offers several efficient implementations of popular LLMs (such as OPT and LLaMA) using PyTorch, CUDA, and Triton, including a special [customized kernel](shared_attention_kernel.md) for our scenarios.

## Server

Each Builtin Engine has a HTTP server to serve the [internal APIs](engine_apis.md) between Engine Layer and Serve Layer. The engine serve has a `engine_loop` which runs forever. In each iteration, the engine will execute a batching inference of the LLM.

## Runner

The Runner inside the engine provides a minimal interface to run the LLM: `run_iter(jobs: List[PrimitiveJob])`. The runner itself does not maintain any immediate states. Instead, it just runs a batch of jobs (primitive requests) and the updates will be recorded in the `PrimitiveJob` objects.

## Memory

### Model Weights

Parrot Builtin Engine directly loads the full model weights into GPU global memory for now. See `parrot/engine/builtin/model_instantiation.py`.

### KV Cache

Parrot Builtin Engine employs [PagedAttention](https://arxiv.org/abs/2309.06180) to divide KV Cache into blocks, storing them in the GPU’s global memory. Parrot supports different kinds of Memory layouts, such as normal layout (K and V are the same), TokenAttention (`block_size=1`), vLLM-style (The `hidden_size` dimension of the K cache is split by a constant `x` for better memory access).


### Cos/Sin Cache

For some models with RoPE (e.g. LLaMA), there is a special cache we can maintain: Cos/Sin Cache. For a given `max_seq_len` of the model, we can pre-compute the value of cos/sin in the RoPE algorithm and reuse them in every subsequent forward pass.