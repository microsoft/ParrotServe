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

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer


# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


# Parrot VM
import parrot as P


def get_chunks(file_name: str, chunk_size: int):
    loader = TextLoader(f"../workloads/arxiv-march-2023/arxiv-sampled/{file_name}.txt")
    docs = loader.load()

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=0,
        separator=" ",
    )
    split_docs = text_splitter.split_documents(docs)

    return [doc.page_content for doc in split_docs]


def get_map_reduce_functions(vm: P.VirtualMachine, chunk_num: int, output_len: int):
    map_func = vm.define_function(
        func_name=None,
        func_body="""Write an one-sentence summary (AS SHORT AS POSSIBLE) of the following:
{{text}}
CONCISE SUMMARY:{{summary}}""",
        cache_prefix=False,
        params=[
            P.Parameter(name="text", typ=P.ParamType.INPUT_LOC),
            P.Parameter(
                name="summary",
                typ=P.ParamType.OUTPUT_LOC,
                sampling_config=P.SamplingConfig(
                    ignore_tokenizer_eos=True,
                    max_gen_length=output_len,
                ),
            ),
        ],
    )

    docs = ["{{" + f"chunk_{i}" + "}}" for i in range(chunk_num)]
    docs = "\n".join(docs)

    reduce_template = (
        "The following is set of summaries:"
        f"{docs}"
        "Take these and distill it into a final, consolidated summary of the main themes."
        "Helpful Answer: {{summary}}"
    )

    input_params = [
        P.Parameter(name=f"chunk_{i}", typ=P.ParamType.INPUT_LOC)
        for i in range(chunk_num)
    ]

    output_param = P.Parameter(
        name="summary",
        typ=P.ParamType.OUTPUT_LOC,
        sampling_config=P.SamplingConfig(
            ignore_tokenizer_eos=True,
            max_gen_length=output_len,
        ),
        dispatch_annotation=P.DispatchAnnotation(
            requests_num_upperbound=32,
        )
    )

    reduce_func = vm.define_function(
        func_name=None,
        func_body=reduce_template,
        cache_prefix=False,
        params=input_params + [output_param],
    )

    return map_func, reduce_func


def sample_requests(
    num_apps: int,
) -> List[int]: # article_no
    return [(0, 0.5), (0, 0.5), (0, 10), (0, 0.5), (0, 0.5), (0, 10), (0, 0.5), (0, 0.5), (0, 10)]


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    app_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request[0]

        if app_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        await asyncio.sleep(request[1])


async def send_request(
    vm: P.VirtualMachine,
    article_no: int
) -> None:
    global REQUEST_LATENCY

    chunk_size = 1024
    output_length = 50

    file_name = f"article_{article_no}"
    chunks = get_chunks(file_name, chunk_size)
    chunk_num = len(chunks)
    map_func, reduce_func = get_map_reduce_functions(vm, chunk_num, output_length)
   
    docs = [P.variable(name=f"output_{i}") for i in range(chunk_num)]
    coros = []
    for i, chunk in enumerate(chunks):
        coros.append(map_func.ainvoke(text=chunk, summary=docs[i]))
    
    request_start_time = time.perf_counter_ns()
    await asyncio.gather(*coros)
    output = await reduce_func.ainvoke(*docs)
    await output.aget()
    request_end_time = time.perf_counter_ns()

    request_latency = (request_end_time - request_start_time) / 1e6
    REQUEST_LATENCY.append((article_no, output_length, request_latency))


async def benchmark(
    vm: P.VirtualMachine,
    input_requests: List[int],
    app_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for article_no in get_request(input_requests, app_rate):
        task = asyncio.create_task(
            send_request(vm, article_no)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_requests = sample_requests(args.num_apps)

    vm = P.VirtualMachine(os_http_addr="http://localhost:9000", mode="debug")
    vm.set_global_env()

    benchmark_start_time = time.perf_counter_ns()

    asyncio.run(
        benchmark(
            vm,
            input_requests,
            args.app_rate,
        )
    )

    benchmark_end_time = time.perf_counter_ns()

    global REQUEST_LATENCY

    benchmark_time = (benchmark_end_time - benchmark_start_time) / 1e6
    print(f"Total time: {benchmark_time / 1e3:.2f} s")
    print(f"Throughput: {args.num_apps * 1e3 / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--num-apps", type=int, default=1000, help="Number of MR apps to process."
    )
    parser.add_argument(
        "--app-rate",
        type=float,
        default=float("inf")
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
