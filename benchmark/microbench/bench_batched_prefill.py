from parrot.engine.builtin.native_runner import NativeRunner
from parrot.engine.config import BuiltinConfig
from parrot.engine.primitive_job import Fill, Generate
from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import torch_profile

import torch
import time


base = 0


def _init():
    global base

    config = BuiltinConfig(
        num_kv_cache_blocks=1000,
        attn_func="xformers_fill_vllm_paged_attention_generate",
        block_size=16,
    )

    runner = NativeRunner(model_name="lmsys/vicuna-7b-v1.3", config=config)

    prompt_len = 670
    fill_num = 10

    fills = [
        Fill(
            pid=i + base,
            tid=i + base,
            context_id=i + base,
            parent_context_id=-1,
            token_ids=[100] * prompt_len,
        )
        for i in range(fill_num)
    ]

    base += fill_num

    return runner, fills


@torch.inference_mode()
def bench_one() -> float:
    runner, fills = _init()

    torch.cuda.synchronize()
    st = time.perf_counter_ns()

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ]
    # ) as p:
    runner.run_iter(fills)
    # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    return (ed - st) / 1e6


def bench_7b_model():
    warmups = 1
    trials = 1

    for _ in range(warmups):
        bench_one()

    total_time = 0
    for _ in range(trials):
        total_time += bench_one()

    print(f"Time: {total_time / trials:.2f} ms")


if __name__ == "__main__":
    bench_7b_model()
