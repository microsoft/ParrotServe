from parrot.utils import torch_profile
from parrot.testing.vllm_runner import vLLMRunner
import time
import torch


def bench_7b_model():
    runner = vLLMRunner(model="lmsys/vicuna-7b-v1.3", max_tokens_sum=81000)

    runner.prefill_random_data(1, 1024, 200)

    warmups = 10
    trials = 100

    for _ in range(warmups):
        runner.step()

    torch.cuda.synchronize()
    st = time.perf_counter_ns()

    for _ in range(trials):
        runner.step()

    torch.cuda.synchronize()
    ed = time.perf_counter_ns()

    print(f"Per decode time: {(ed - st) / 1e6 / trials} ms")


def profile_7b_model():
    runner = vLLMRunner(model="lmsys/vicuna-7b-v1.3", max_tokens_sum=81000)

    runner.prefill_random_data(1, 1024, 200)

    warmups = 10

    for _ in range(warmups):
        runner.step()

    with torch_profile("7b_model"):
        runner.step()


if __name__ == "__main__":
    bench_7b_model()
    # profile_7b_model()
