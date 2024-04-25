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

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer


# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []



def prepare_chains(output_len: int):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=output_len)

    map_inst = """Write an one-sentence summary (AS SHORT AS POSSIBLE) of the following:
    {text}
    CONCISE SUMMARY:"""
    map_template = PromptTemplate(
        input_variables=["text"],
        template=map_inst,
    )
    map_chain = LLMChain(llm=llm, prompt=map_template)

    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate(input_variables=["docs"], template=reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    return map_chain, reduce_chain


def prepare_docs(file_name: str, chunk_size: int):
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
    return split_docs


async def amap(map_chain, doc: str):
    resp = await map_chain.arun(text=doc)
    return resp


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
    article_no: int
) -> None:
    global REQUEST_LATENCY

    chunk_size = 1024
    output_length = 50

    file_name = f"article_{article_no}"
    map_chain, reduce_chain = prepare_chains(output_length)
    split_docs = prepare_docs(file_name, chunk_size)
    coros = []
    for doc in split_docs:
        coros.append(amap(map_chain=map_chain, doc=doc.page_content))

    request_start_time = time.perf_counter_ns()
    docs = await asyncio.gather(*coros)
    docs = "\n".join(docs[:4000]) # prevent stuck
    resp = await reduce_chain.arun(docs=docs)  # This is to avoid stuck

    request_end_time = time.perf_counter_ns()

    request_latency = (request_end_time - request_start_time) / 1e6
    REQUEST_LATENCY.append((article_no, output_length, request_latency))


async def benchmark(
    input_requests: List[int],
    app_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for article_no in get_request(input_requests, app_rate):
        task = asyncio.create_task(
            send_request(article_no)
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_requests = sample_requests(args.num_apps)

    benchmark_start_time = time.perf_counter_ns()

    asyncio.run(
        benchmark(
            input_requests,
            args.app_rate,
        )
    )

    benchmark_end_time = time.perf_counter_ns()

    global REQUEST_LATENCY

    benchmark_time = (benchmark_end_time - benchmark_start_time) / 1e6
    print(f"Total time: {benchmark_time / 1e3:.2f} s", flush=True)
    print(f"Throughput: {args.num_apps * 1e3 / benchmark_time:.2f} requests/s", flush=True)

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} ms", flush=True)


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
