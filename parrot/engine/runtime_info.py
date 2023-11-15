# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass


@dataclass
class EngineRuntimeInfo:
    num_cached_tokens: int = 0
    num_running_jobs: int = 0

    # NOTE(chaofan): All memory fields are in MiB.
    cache_mem: float = 0
    model_mem: float = 0

    recent_latency: float = 0  # in nanoseconds
