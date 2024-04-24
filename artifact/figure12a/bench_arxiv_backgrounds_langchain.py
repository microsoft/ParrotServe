# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time
import multiprocessing as mp
import numpy as np
import asyncio
import sys


### Langchain part

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from transformers import AutoTokenizer


def proc1(barrier: mp.Barrier, file_name: str):
    chunk_size = 1024
    output_len = 50

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=output_len)
    chain = load_summarize_chain(llm, chain_type="refine")

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

    barrier.wait()

    time.sleep(3)

    for _ in range(1):
        st = time.perf_counter_ns()
        chain.run(split_docs)
        ed = time.perf_counter_ns()
        print(f"Time: {(ed - st) / 1e9:.4f} s", flush=True)
        # time.sleep(3)


def proc2(barrier: mp.Barrier, request_rate: float):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=50)

    async def _generator():
        while True:
            yield "Test " * 512
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)

    async def _proc2():
        tasks = []

        if request_rate == 0:
            return

        async for request in _generator():
            task = asyncio.create_task(llm.ainvoke(request))
            tasks.append(task)

        await asyncio.gather(*tasks)

    barrier.wait()
    asyncio.run(_proc2())


# print(llm.invoke("Hello!"))


def main(file_name: str, request_rate: float):
    print(f"file_name: {file_name}, request_rate: {request_rate}", flush=True)

    barrier = mp.Barrier(2)

    p1 = mp.Process(target=proc1, args=(barrier, file_name))
    p2 = mp.Process(target=proc2, args=(barrier, request_rate))

    p1.start()
    p2.start()

    p1.join()
    p2.terminate()  # Directly shutdown p2


def warmup():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=5)
    llm.invoke("hello")


if __name__ == "__main__":
    # main("article_5", 3)
    # for i in range(6, 10):
    #     for reqs in [1, 2, 3]:
    #         main(f"article_{i}", reqs)
    #         time.sleep(10)

    warmup()

    article_id = int(sys.argv[1])

    for reqs in [0, 1, 2, 3, 3.5]:
        main(f"article_{article_id}", reqs)
