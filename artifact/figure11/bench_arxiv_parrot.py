# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import sys
import parrot as P

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


def get_refine_functions(file_name: str, chunk_num: int, output_len: int):
    global vm

    rets = []

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

    rets.append(first_func)

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

    for i in range(1, chunk_num):
        func = vm.define_function(
            func_name=f"refine_func_{i}",
            func_body=refine_template,
            cache_prefix=True,
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


def main(file_name: str, chunk_size: int, output_len: int):
    chunks = get_chunks(file_name, chunk_size)
    funcs = get_refine_functions(file_name, len(chunks), output_len)

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


def warmup():
    global vm
    test_func = vm.import_function(
        "func_1i_1o_genlen_100", "artifact.workloads.test_examples.normal_functions"
    )
    with vm.running_scope():
        holder = test_func("Test")
        holder.get()


if __name__ == "__main__":
    warmup()

    arg = sys.argv[1]

    if arg == "test":
        main("article_8", 1024, 1)
    elif arg == "exp1":
        for i in range(10):
            for ol in [25, 50, 75, 100]:
                main(f"article_{i}", 1024, ol)
    elif arg == "exp2":
        for i in range(10):
            for bs in [512, 1024, 1536, 2048]:
                main(f"article_{i}", bs, 50)
