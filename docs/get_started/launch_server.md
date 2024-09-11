# Launch a Parrot Server

This document will guide you to launch a Parrot Server and have a basic understanding of how to configure your server.

## Overview of Parrot Architecture

To better understand how to launch a Parrot Server, it's necessary to introduce the architecture of Parrot first. Note that it's just a brief introduction. To get more specific information, see [Parrot System Design](../sys_design/).

Parrot is a distributed serving system which contains a centralized `ServeCore` and several `Engine`s connected to the core. `ServeCore` communicates with`Engine`s through HTTP and manages them through a mechanism of registration and heartbeat packets, allowing `Engine` to plugin in/out of the system flexibly.

Hence, the recommended launch process of a Parrot Server is:
- Launch a `ServeCore` server.
- Launch several `Engine`s, waiting for them to be registered in the `ServeCore`.
- If you want to down a `Engine`, just kill the corresponding process. It will be automatically unregistered when the heartbeat is expired.

## Launch the `ServeCore`

You can start a `ServeCore` server through the following command.

```bash
python3 -m parrot.serve.http_server --config_path <config_path> \
    --log_dir <log_dir> \
    --log_filename <log_filename>
```

For other command line arguments, run
```bash
python3 -m parrot.serve.http_server --help
```

## Launch an `Engine`

You can separately start an `Engine` server. If you choose to connect to the `ServeCore` server, you need to start the `ServeCore` server first and specify the `ServeCore` server address in the config file.

```bash
python3 -m parrot.engine.http_server --config_path <config_path> \
    --log_dir <log_dir> \
    --log_filename <log_filename>
```

For other command line arguments, run
```bash
python3 -m parrot.engine.http_server --help
```

## Config Files Specification

We put some sample config files under `sample_configs/core/` (For `ServeCore`) and `sample_configs/engine/` (For `Engine`).

We use `.json` as the format of our configuration file.

**`ServeCore` Config**

For detailed features of global scheduler, please refer to [Global Scheduler](../sys_design/serve_layer/global_scheduler.md).

```json
{
    "host": "localhost", // Host of the server.
    "port": 9000, // Port of the server.
    "max_sessions_num": 2048, // Max number of sessions ServeCore can manage.
    "max_engines_num": 2048, // Max number of engines ServeCore can manage.
    "session_life_span": 9999999, // Life span (expiration time) of a session.
    "global_scheduler": { // Config of the global scheduler.
        "app_fifo": false, // Turn on/off the app fifo.
        "graph_group": false, // Turn on/off the graph group.
        "ctx_group": false, // Turn on/off the context group.
        "ctx_aware": false, // Turn on/off the context-aware scheduling.
        "max_queue_size": 2048 // Max queue size of scheduler.
    }
}
```

**`Engine` Config for Local LLMs**

```json
{
    "engine_name": "vicuna-7b-v1.3_local", // Engine name. Just for managing and debugging.
    "model": "lmsys/vicuna-7b-v1.3", // Model name. It should be consist with the hugging face model name.
    "host": "localhost", // Host of the server.
    "port": 9001, // Port of the server.
    "engine_type": "builtin", // Engine type. For local LLMs, choose "builtin".
    "random_seed": 0, // Random seed.
    "tokenizer": "hf-internal-testing/llama-tokenizer", // Tokenizer name. It should be consist with the hugging face tokenizer name.
    "fill_chunk_size": -1, // Chunked prefill size. -1: no chunked.
    "tasks_capacity": 256, // Capacity of tasks in this engine.
    "instance": { // Config of the instance. For builtin instances, we need to specify numbers of KV Cache blocks, the attention function we use, etc. For more information, see parrot/engine/config.py.
        "num_kv_cache_blocks": 8000,
        "attn_func": "xformers_with_buffer"
    },
    "scheduler": { // Config of the local scheduler.
        "max_batch_size": 256,
        "max_num_batched_tokens": 2560,
        "max_total_tokens": 8192
    },
    "serve_core": { // Config of the ServeCore this engine should connect to.
        "host": "localhost",
        "port": 9000
    }
}
```

**`Engine` Config for Azure OpenAI**

```json
{
    "model": "gpt35turobo4k", // Your deployment name
    "engine_name": "Azure-OpenAI-GPT-3.5-Turbo-4K", // Engine name. Just for managing and debugging.
    "host": "localhost", // Host of the server.
    "port": 9001, // Port of the server.
    "engine_type": "openai", // Engine type. Use `openai` here.
    "random_seed": 0, // Random seed.
    "tasks_capacity": 64, // Capacity of tasks in this engine.
    "instance": { // Config of the instance. For AzureOpenAI instance, we need to specify its api_key, api_version, azure_endpoint, etc. And most importantly, set is_azure=true. For more information, see parrot/engine/config.py.
        "api_key": "xxx",
        "api_endpoint": "completion",
        "is_azure": true,
        "azure_api_version": "2023-07-01-preview",
        "azure_endpoint": "xxx"
    },
    "scheduler": { // Config of the local scheduler. For OpenAI instance, local scheduler is not useful since all requests will be finally sent to OpenAI API and scheduled by cloud side.
        "max_batch_size": 256,
        "max_num_batched_tokens": 99999999,
        "max_total_tokens": 99999999
    },
    "serve_core": { // Config of the ServeCore this engine should connect to.
        "host": "localhost",
        "port": 9000
    }
}
```