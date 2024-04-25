# Modified from:
# https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py

"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple, Optional

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


# Parrot VM
import parrot as P
from parrot.utils import cprofile

vm: Optional[P.VirtualMachine] = None


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        # yield ("Test " * 512, 512, 50)
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        # interval = np.random.exponential(1.0 / request_rate)
        interval = 1.0 / request_rate  # uniform
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


req_counter = 0


async def send_request(
    prompt: str,
    prompt_len: int,
    output_len: int,
) -> None:
    global REQUEST_LATENCY
    global req_counter
    req_no = req_counter
    req_counter += 1

    func = vm.define_function(
        func_name="chat",
        func_body="{{input}}{{output}}",
        params=[
            P.Parameter("input", P.ParamType.INPUT_LOC),
            P.Parameter(
                "output",
                P.ParamType.OUTPUT_LOC,
                sampling_config=P.SamplingConfig(
                    max_gen_length=output_len,
                    ignore_tokenizer_eos=True,
                ),
                dispatch_annotation=P.DispatchAnnotation(
                    requests_num_upperbound=16,
                ),
            ),
        ],
        cache_prefix=False,
    )

    request_start_time = time.perf_counter_ns()

    output = await func.ainvoke(f"chaos#%{req_no}chaos#%" + prompt)
    await output.aget()

    request_end_time = time.perf_counter_ns()
    request_latency = (request_end_time - request_start_time) / 1e6
    REQUEST_LATENCY.append((req_no, prompt_len, output_len, request_latency))


async def benchmark(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(
                prompt,
                prompt_len,
                output_len,
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    global vm
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000", mode="debug")
    vm.set_global_env()

    benchmark_start_time = time.perf_counter_ns()

    asyncio.run(
        benchmark(
            input_requests,
            args.request_rate,
        )
    )

    global REQUEST_LATENCY

    benchmark_end_time = time.perf_counter_ns()
    benchmark_time = (benchmark_end_time - benchmark_start_time) / 1e9
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    for req_no, prompt_len, output_len, request_latency in REQUEST_LATENCY:
        print(
            f"Request {req_no}: latency={request_latency:.2f} ms, output_len={output_len}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    args = parser.parse_args()
    main(args)
