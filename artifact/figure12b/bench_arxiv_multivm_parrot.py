# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from numpy import mean
import time
import parrot as P
from multiprocessing import Barrier
from parrot.testing.multiproc_manager import MultiProcessManager
from parrot.utils import cprofile


def get_chunks(file_name: str, chunk_size: int):
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from transformers import AutoTokenizer

    loader = TextLoader(
        f"../workloads/arxiv-march-2023/arxiv-sampled-1/{file_name}.txt"
    )
    docs = loader.load()

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=0,
        separator=" ",
    )
    split_docs = text_splitter.split_documents(docs)

    # for i, doc in enumerate(split_docs):
    #     print(i, len(tokenizer.encode(doc.page_content)))

    # 0 / 0

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


def process(barrier: Barrier, article_no: int):
    chunk_size = 2048
    output_len = 50

    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    chunks = get_chunks(f"article_{article_no}", chunk_size)
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

    latency = vm.run(main_async, timeit=True)
    # print(f"Time: {latency:.4f}", flush=True)
    # time.sleep(3)

    return latency


def main(clients_num: int):
    # print("chunk_size:", chunk_size, flush=True)
    print("clients_num:", clients_num, flush=True)
    # clients_num = 8

    manager = MultiProcessManager()
    barrier = Barrier(clients_num)

    for i in range(clients_num):
        manager.add_proc(process, (barrier, i))

    manager.run_all()
    print(manager.data)
    print(f"Avg. JCT {mean(list(manager.data.values())):.2f} (s)")


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

    # test_baseline()
    # main(10)
    # main(4)
    # time.sleep(10)
    # main(8)
    # time.sleep(10)
    # main(20)
    for num in [10, 15, 20, 25]:
        main(num)
    # main(30)
