# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import asyncio
import parrot as P
import parse

vm = P.VirtualMachine(os_http_addr="http://localhost:9000")


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


def get_map_reduce_functions(file_name: str, chunk_num: int, output_len: int):
    global vm

    rets = []

    map_func = vm.define_function(
        func_name="map_func",
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

    reduce_template = (
        "The following is set of summaries:"
        f"{docs}"
        "Take these and distill it into a final, consolidated summary of the main themes."
        "Helpful Answer:"
    )

    input_params = [
        P.Parameter(name=f"chunk_{i}", typ=P.ParamType.INPUT_LOC)
        for i in range(chunk_num)
    ]

    output_param = (
        P.Parameter(
            name="summary",
            typ=P.ParamType.OUTPUT_LOC,
            sampling_config=P.SamplingConfig(
                ignore_tokenizer_eos=True,
                max_gen_length=output_len,
            ),
        ),
    )

    reduce_func = vm.define_function(
        func_name="reduce_func",
        func_body=reduce_template,
        cache_prefix=False,
        params=input_params
        + [
            output_param,
        ],
    )

    return map_func, reduce_func


def main(file_name: str, chunk_size: int, output_len: int):
    chunks = get_chunks(file_name, chunk_size)
    funcs = get_map_reduce_functions(file_name, len(chunks), output_len)

    print(
        f"file_name: {file_name}, chunk_size: {chunk_size}, output_len: {output_len}",
        flush=True,
    )

    async def _main():
        # NOTE(chaofan): We only get the final result, let the intermediate results be
        # flown in the system.

        next_input = funcs[0](text=chunks[0])

        for func, chunk in zip(funcs[1:], chunks[1:]):
            next_input = func(existing_answer=next_input, text=chunk)

        next_input.get()

    for _ in range(1):
        latency = vm.run(_main, timeit=True)
        print(f"Time: {latency:.4f}", flush=True)
        time.sleep(3)


def warmup():
    global vm
    test_func = vm.import_function(
        "chain_sum_test", "benchmark.workloads.test_examples.chain_summarization"
    )
    with vm.running_scope():
        holder = test_func("Test " * 100)
        holder.get()


if __name__ == "__main__":
    warmup()

    print("warmup done", flush=True)

    # for i in range(10):
    #     for ol in [25, 50, 75, 100]:
    #         main(f"article_{i}", 1024, ol)

    for i in range(8, 10):
        for bs in [512, 1024, 1536, 2048]:
            main(f"article_{i}", bs, 50)
