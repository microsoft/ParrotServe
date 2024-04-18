# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time
import multiprocessing as mp
import numpy as np
import asyncio

module = importlib.import_module(
    f"benchmark.workloads.test_examples.chain_summarization"
)
fake_long_document_chunk = getattr(module, "fake_long_document_chunk")

chunk_num = 20

full_document = (fake_long_document_chunk + "\n\n") * chunk_num

with open("test.txt", "w") as f:
    f.write(full_document)


### Langchain part

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


def proc1(barrier: mp.Barrier):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=20)
    chain = load_summarize_chain(llm, chain_type="refine")

    loader = TextLoader("test.txt")
    docs = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=650,
        chunk_overlap=0,
    )
    split_docs = text_splitter.split_documents(docs)

    for i, doc in enumerate(split_docs):
        print(i, len(doc.page_content.split(" ")))

    barrier.wait()

    st = time.perf_counter_ns()
    chain.run(split_docs)
    ed = time.perf_counter_ns()
    with open("langchain_stdout.log", "a+") as f:
        print(f"Time: {(ed - st) / 1e9} s", file=f, flush=True)
    time.sleep(3)


def proc2(barrier: mp.Barrier, request_rate: float):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=100)

    requests_num = 1000

    async def _generator():
        for _ in range(requests_num):
            yield "Test " * 100
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)

    async def _proc2():
        tasks = []

        async for request in _generator():
            task = asyncio.create_task(llm.ainvoke(request))
            tasks.append(task)

        await asyncio.gather(*tasks)

    barrier.wait()
    asyncio.run(_proc2())


# print(llm.invoke("Hello!"))


def main(request_rate: float):
    barrier = mp.Barrier(2)

    p1 = mp.Process(target=proc1, args=(barrier,))
    p2 = mp.Process(target=proc2, args=(barrier, request_rate))

    p1.start()
    p2.start()

    p1.join()
    p2.terminate()  # Directly shutdown p2


if __name__ == "__main__":
    main(2.0)
