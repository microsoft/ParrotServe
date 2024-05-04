# Modified from:
# https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py

import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple, Optional, Dict

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def get_func(prompt: str, output_len: int):
    async def invoke_chain(query: str):
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            max_tokens=output_len,
        )
        await llm.ainvoke(prompt + query)

    return invoke_chain


def define_functions(workload_info_path: str):
    # Load the dataset.
    with open(workload_info_path) as f:
        workload_info = json.load(f)

    funcs = {}

    # Define the functions.
    for app_info in workload_info:
        app_name = app_info["app_name"]
        prompt_length = app_info["prompt_length"]
        output_length = app_info["output_length"]

        prompt = " ".join(["Test"] * prompt_length)

        funcs[app_name] = get_func(prompt, output_length)

    return funcs


def sample_requests(
    workload_info_path: str,
    num_requests: int,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(workload_info_path) as f:
        workload_info = json.load(f)

    dataset = []
    # total_requests = 10000
    total_requests = num_requests  # for stable reproduction
    for app_info in workload_info:
        # total_requests * app_info["percentage"]
        app_num_reqs = total_requests / len(
            workload_info
        )  # 1:1:1:1 instead of percentage
        for _ in range(int(app_num_reqs)):
            app_name = app_info["app_name"]
            query_length = app_info["query_length"]
            output_length = app_info["output_length"]
            dataset.append((app_name, query_length, output_length))

    # Sample the requests.
    # sampled_requests = random.sample(dataset, num_requests)
    random.shuffle(dataset)
    sampled_requests = dataset
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    funcs: Dict,
    app_name: str,
    query_length: int,
    output_length: int,
) -> None:
    global REQUEST_LATENCY

    func = funcs[app_name]
    query = " ".join(["Test"] * query_length)

    request_start_time = time.perf_counter_ns()
    # Send the request.
    await func(query)
    request_end_time = time.perf_counter_ns()

    request_latency = (request_end_time - request_start_time) / 1e6
    REQUEST_LATENCY.append((app_name, output_length, request_latency))


async def benchmark(
    funcs: Dict,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        app_name, query_len, output_len = request
        task = asyncio.create_task(
            send_request(
                funcs,
                app_name,
                query_len,
                output_len,
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    # print(args)
    print("request_rate: ", args.request_rate)
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_requests = sample_requests(args.workload_info, args.num_prompts)

    funcs = define_functions(args.workload_info)
    benchmark_start_time = time.perf_counter_ns()

    asyncio.run(
        benchmark(
            funcs,
            input_requests,
            args.request_rate,
        )
    )

    benchmark_end_time = time.perf_counter_ns()

    global REQUEST_LATENCY

    benchmark_time = (benchmark_end_time - benchmark_start_time) / 1e6
    # print(f"Total time: {benchmark_time / 1e3:.2f} s")
    # print(f"Throughput: {args.num_prompts * 1e3 / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    # avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    # print(f"Average latency: {avg_latency:.2f} ms")
    avg_per_output_token_latency = np.mean(
        [latency / output_len for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Normalized latency: " f"{avg_per_output_token_latency:.2f} ms")

    # for key in funcs.keys():
    #     print("App name: ", key)
    #     print(f"Number of requests: {len([x for x in REQUEST_LATENCY if x[0] == key])}")
    #     print(f"Average latency: {np.mean([x[2] for x in REQUEST_LATENCY if x[0] == key]):.2f} ms")
    #     print(f"Average latency per output token: {np.mean([x[2] / x[1] for x in REQUEST_LATENCY if x[0] == key]):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--workload-info", type=str, required=True, help="Path to the workload info."
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
    args = parser.parse_args()
    main(args)
