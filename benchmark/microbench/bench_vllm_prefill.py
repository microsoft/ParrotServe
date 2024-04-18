from parrot.utils import torch_profile
from parrot.testing.vllm_runner import vLLMRunner
import time
import torch


def bench_7b_model():
    runner = vLLMRunner(model="lmsys/vicuna-7b-v1.3", max_tokens_sum=81000)

    warmups = 1
    trials = 1

    for _ in range(warmups):
        runner.prefill_random_data(20, 670, 2)

    torch.cuda.synchronize()
    st = time.perf_counter_ns()

    for _ in range(trials):
        runner.prefill_random_data(20, 670, 2)

    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(f"Per decode time: {(ed - st) / 1e6 / trials} ms")


if __name__ == "__main__":
    bench_7b_model()
    # profile_7b_model()
