from parrot.engine.builtin.builtin_runner import BuiltinRunner
from parrot.engine.config import BuiltinConfig
from parrot.engine.primitive_job import Fill, Generate
from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import torch_profile, cprofile

import torch
import time


def _init(model):
    config = BuiltinConfig(
        num_kv_cache_blocks=1000,
        attn_func="xformers_fill_vllm_paged_attention_generate",
        block_size=16,
        max_seq_len=4096,
    )

    runner = BuiltinRunner(model_name=model, config=config)

    prompt_len = 3000

    fill1 = Fill(pid=0, tid=0, context_id=0, parent_context_id=-1, token_ids=[100])

    fill2 = Fill(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=[100] * prompt_len,
    )

    sampling_config = SamplingConfig(
        max_gen_length=200,
        ignore_tokenizer_eos=True,
    )

    gen = Generate(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        sampling_config=sampling_config,
    )

    return runner, fill1, fill2, gen


@torch.inference_mode()
def bench_7b_model():
    runner, fill1, fill2, gen = _init("lmsys/vicuna-7b-v1.3")

    warmups = 10
    trials = 100

    runner.run_iter([fill1])
    runner.run_iter([fill2])

    for _ in range(warmups):
        runner.run_iter([gen])

    torch.cuda.synchronize()
    st = time.perf_counter_ns()

    for _ in range(trials):
        runner.run_iter([gen])

    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(f"Per decode time: {(ed - st) / 1e6 / trials} ms")


@torch.inference_mode()
def bench_13b_model():
    runner, fill1, fill2, gen = _init("lmsys/vicuna-13b-v1.3")

    warmups = 10
    trials = 100

    runner.run_iter([fill1])
    runner.run_iter([fill2])

    for _ in range(warmups):
        runner.run_iter([gen])

    torch.cuda.synchronize()
    st = time.perf_counter_ns()

    for _ in range(trials):
        runner.run_iter([gen])

    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(f"Per decode time: {(ed - st) / 1e6 / trials} ms")


def profile_7b_model():
    runner, fill1, fill2, gen = _init("lmsys/vicuna-7b-v1.3")

    runner.run_iter([fill1])
    runner.run_iter([fill2])

    warmups = 10

    for _ in range(warmups):
        runner.run_iter([gen])

    # with torch_profile("7b_model"):
    for i in range(10):
        with cprofile(f"iter_{i}"):
            runner.run_iter([gen])


def profile_13b_model():
    runner, fill1, fill2, gen = _init("lmsys/vicuna-13b-v1.3")

    runner.run_iter([fill1])
    runner.run_iter([fill2])

    warmups = 10

    for _ in range(warmups):
        runner.run_iter([gen])

    with torch_profile("13b_model"):
        for i in range(10):
            # with cprofile(f"iter_{i}"):
            runner.run_iter([gen])


if __name__ == "__main__":
    # bench_7b_model()
    bench_13b_model()
    # profile_7b_model()
    # profile_13b_model()
