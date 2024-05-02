# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import asyncio
import multiprocessing as mp
import parrot as P
from parrot.utils import cprofile
import numpy as np


def get_chunks(file_name: str, chunk_size: int):
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from transformers import AutoTokenizer

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


def get_functions(vm: P.VirtualMachine, output_len: int):
    first_func = vm.define_function(
        func_name="first_func",
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

    refine_template = (
        "Your job is to produce an one-sentence summary (AS SHORT AS POSSIBLE) for a long document.\n"
        "We have provided an existing summary up to a certain point: {{existing_answer}}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{{text}}\n"
        "------------\n"
        "Given the new context, refine the original summary in English. "
        "If the context isn't useful, return the original summary.\n"
        "{{summary}}"
    )

    func = vm.define_function(
        func_name=f"refine_func",
        func_body=refine_template,
        cache_prefix=False,
        params=[
            P.Parameter(name="existing_answer", typ=P.ParamType.INPUT_LOC),
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

    return first_func, func


def proc1(barrier: mp.Barrier, file_name: str):
    chunk_size = 1024
    output_len = 50

    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    chunks = get_chunks(file_name, chunk_size)
    chunk_num = len(chunks)
    func1, func2 = get_functions(vm, output_len)

    async def main_async():
        outputs = [P.variable(name=f"output_{i}") for i in range(chunk_num)]
        vm.set_batch()
        for i in range(chunk_num):
            if i == 0:
                func1(text=chunks[0], summary=outputs[i])
            else:
                func2(
                    existing_answer=outputs[i - 1],
                    text=chunks[i],
                    summary=outputs[i],
                )
        await vm.submit_batch()
        outputs[-1].get()

    barrier.wait()

    # Wait for entering
    time.sleep(3)

    for _ in range(1):
        latency = vm.run(main_async, timeit=True)
        print(f"Time: {latency:.4f}", flush=True)
        # time.sleep(3)


def proc2(barrier: mp.Barrier, request_rate: float):
    if request_rate == 0:
        barrier.wait()
        return

    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    test_func = vm.define_function(
        func_name="test_func",
        func_body="Test " * 512 + "{{output}}",
        params=[
            P.Parameter(
                name="output",
                typ=P.ParamType.OUTPUT_LOC,
                sampling_config=P.SamplingConfig(
                    max_gen_length=50, ignore_tokenizer_eos=True
                ),
            )
        ],
    )

    outputs = []

    with vm.running_scope():
        barrier.wait()

        while True:
            output = test_func()
            outputs.append(output)
            interval = np.random.exponential(1.0 / request_rate)
            time.sleep(interval)


def main(file_name: str, request_rate: float):
    print(f"file_name: {file_name}, request_rate: {request_rate}", flush=True)

    barrier = mp.Barrier(2)
    p1 = mp.Process(
        target=proc1,
        args=(barrier, file_name),
    )
    p2 = mp.Process(
        target=proc2,
        args=(
            barrier,
            request_rate,
        ),
    )

    p1.start()
    p2.start()

    p1.join()
    p2.terminate()  # Directly shutdown p2


def warmup():
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")
    test_func = vm.import_function(
        "func_1i_1o_genlen_100", "artifact.workloads.test_examples.normal_functions"
    )
    with vm.running_scope():
        holder = test_func("Test")
        holder.get()


if __name__ == "__main__":
    warmup()

    for i in range(10):
        for reqs in [0, 1, 2, 3, 3.5]:
            main(f"article_{i}", reqs)
