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


def get_map_reduce_functions(file_name: str, chunk_num: int, output_len: int):
    global vm

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
    )

    reduce_func = vm.define_function(
        func_name="reduce_func",
        func_body=reduce_template,
        cache_prefix=False,
        params=input_params + [output_param],
    )

    return map_func, reduce_func


def main(file_name: str, chunk_size: int, output_len: int):
    chunks = get_chunks(file_name, chunk_size)
    chunk_num = len(chunks)
    map_func, reduce_func = get_map_reduce_functions(file_name, chunk_num, output_len)

    print(
        f"file_name: {file_name}, chunk_size: {chunk_size}, output_len: {output_len}",
        flush=True,
    )

    async def _main():
        vm.set_batch()

        docs = [P.variable(name=f"output_{i}") for i in range(chunk_num)]
        for i, chunk in enumerate(chunks):
            map_func(text=chunk, summary=docs[i])

        await vm.submit_batch()
        output = reduce_func(*docs)
        output.get()

    for _ in range(1):
        latency = vm.run(_main, timeit=True)
        print(f"Time: {latency:.4f}", flush=True)
        # time.sleep(10)


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

    # main("article_0", 1024, 25)

    # main("article_0", 1024, 50)

    # main("article_6", 1024, 75)

    arg = sys.argv[1]

    if arg == "test":
        main("article_0", 1024, 100)
    elif arg == "exp1":
        for i in range(10):
            for ol in [25, 50, 75, 100]:
                main(f"article_{i}", 1024, ol)
    elif arg == "exp2":
        for i in range(10):
            for cs in [512, 1024, 1536, 2048]:
                main(f"article_{i}", cs, 50)
