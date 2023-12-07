# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass


@dataclass
class VMRuntimeInfo:
    """Runtime information of a VM."""

    mem_used: float = 0
    num_total_tokens: int = 0
    num_threads: int = 0

    def display(self) -> str:
        ret = ""
        for key, value in self.__dict__.items():
            if "latency" in key:
                ret += f"\t{key}: {value / 1e6:.3f} ms\n"
            elif "mem" in key:
                ret += f"\t{key}: {value:.3f} MiB\n"
            else:
                ret += f"\t{key}: {value}\n"
        return ret


@dataclass
class EngineRuntimeInfo:
    """Runtime information of an engine.

    It's the most important message package between OS and engines.

    The size of this package should not be too large, ideally a constant.
    (Won't change with the number of threads and the number of tokens in engine.)

    It appears in two places when the OS and engine communicate:
    - Heartbeat from engine to OS.
    - Ping from OS to engine.

    The heartbeat message package is to maintain a (maybe slightly outdated) view of
    engines in the system. Users in the frontend can use some interfaces to query these
    info, like the number of tokens occupied by certain VM.

    The ping message package is to query a instant runtime information of engines.
    This is necessary for OS to schedule threads.
    """

    num_cached_tokens: int = 0
    num_max_blocks: int = 0
    num_running_jobs: int = 0
    num_total_jobs: int = 0  # Include both running and pending jobs

    # All memory fields are in MiB.
    cache_mem: float = 0
    model_mem: float = 0
    profiled_cpu_mem: float = 0
    profiled_gpu_tensor_mem: float = 0
    profiled_gpu_allocate_mem: float = 0

    # All latency fields are in nanoseconds.
    recent_average_latency: float = 0

    def display(self) -> str:
        ret = ""
        for key, value in self.__dict__.items():
            if "latency" in key:
                ret += f"\t{key}: {value / 1e6:.3f} ms\n"
            elif "mem" in key:
                ret += f"\t{key}: {value:.3f} MiB\n"
            else:
                ret += f"\t{key}: {value}\n"
        return ret
