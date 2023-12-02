# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import asyncio
import parrot as P
import parse

vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

chunk_size = 1024


def get_chunks(file_name: str):
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from transformers import AutoTokenizer

    global chunk_size

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


def get_refine_functions(file_name: str):
    global vm
    global chunk_size

    output_len_file_path = f"../workloads/arxiv-march-2023/arxiv-sampled/chunksize-{chunk_size}/{file_name}-chain-outputlen.txt"
    output_lens = []

    with open(output_len_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            result = parse.parse("Step {step}: Output Len={output_len}", line)
            if result is not None:
                output_lens.append(int(result["output_len"]))

    # print(output_lens)

    rets = []

    first_func = vm.define_function(
        func_name="first_func",
        func_body="""Write a concise summary of the following:
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
                    max_gen_length=output_lens[0],
                ),
            ),
        ],
    )

    rets.append(first_func)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {{existing_answer}}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "!!!IMPORTANT!!! Never let your summary exceeds 50 words.\n"
        "------------\n"
        "{{text}}\n"
        "------------\n"
        "Given the new context, refine the original summary in English. "
        "If the context isn't useful, return the original summary.\n"
        "{{summary}}"
    )

    for i, output_len in enumerate(output_lens[1:]):
        func = vm.define_function(
            func_name=f"refine_func_{i}",
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
        rets.append(func)

    return rets


if __name__ == "__main__":
    chunks = get_chunks("article_0")
    funcs = get_refine_functions("article_0")

    async def main():
        # NOTE(chaofan): We only get the final result, let the intermediate results be
        # flown in the system.

        bs = 8
        next_inputs = [None for _ in range(bs)]

        for i in range(bs):
            next_inputs[i] = funcs[0](text=chunks[0])

        for func, chunk in zip(funcs[1:], chunks[1:]):
            for i in range(bs):
                next_inputs[i] = func(existing_answer=next_inputs[i], text=chunk)

        for i in range(bs):
            next_inputs[i].get()

    vm.run(main, timeit=True)
