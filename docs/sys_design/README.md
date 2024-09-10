# Parrot's System Design

Parrot is a distributed serving system for LLM-based Applications. It can be divided into three layers basically:
- [Application Layer](app_layer/):
    - Parrot's LLM programming frontend: [PFunc](app_layer/pfunc.md).
    - [Semantic Variable](app_layer/semantic_variable.md).
- [Serve Layer](serve_layer/):
    - [ServeCore](serve_layer/core.md), a.k.a. Parrot Manager.
    - [Global Scheduler](serve_layer/global_scheduler.md).
    - [Parrot's Graph Representation](serve_layer/graph.md).
    - [Managers](serve_layer/managers.md).
- [Engine Layer](engine_layer/):
    - [Internal APIs](engine_layer/engine_apis.md) between `ServeCore` and `Engine`.
    - [Builtin Engine](engine_layer/builtin_engine.md).
    - [OpenAI Engine](engine_layer/openai_engine.md).

## Overview

The Parrot API w/ Semantic Variable is served by a centralized cluster manager called `ServeCore`, which manages many `Engine` instances. Each Parrot `Engine` runs a single LLM model and communicates with `ServeCore` by contextual Fill/Gen APIs.

Note that each `Engine` is capable of providing language model services independently, therefore the system is horizontally scalable and many types of `Engine`s can be integrated into Parrot (e.g., vLLM, FasterTransformer, etc.).

The following picture illustrates the overview architecture of Parrot. Please refer our OSDI'24 paper [Parrot: Efficient Serving of LLM-based Applications with Semantic Variable](https://www.usenix.org/system/files/osdi24-lin-chaofan.pdf) for more details.

<div align="center">
  <img src="../images/arch_paper_ver.png" width="500px" />
</div>
