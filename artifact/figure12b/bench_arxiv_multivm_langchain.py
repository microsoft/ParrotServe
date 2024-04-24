# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time
import sys
from numpy import mean
from multiprocessing import Barrier
from parrot.testing.multiproc_manager import MultiProcessManager

from langchain.chat_models import ChatOpenAI


def process(barrier: Barrier, file_name: str):
    chunk_size = 2048
    output_len = 50

    ### Langchain part

    from langchain.chains.summarize import load_summarize_chain
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from transformers import AutoTokenizer

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=output_len)
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

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in English. "
        "If the context isn't useful, return the original summary.\n"
    )

    refine_prompt = PromptTemplate.from_template(refine_template)

    def run_chain():
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            input_key="input_documents",
            output_key="output_text",
        )
        result = chain({"input_documents": split_docs}, return_only_outputs=True)
        return result

    barrier.wait()

    st = time.perf_counter_ns()
    result = run_chain()
    ed = time.perf_counter_ns()

    return (ed - st) / 1e9


def warmup():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=5)
    llm.invoke("hello")


def main(clients_num: int):
    print("clients_num:", clients_num, flush=True)
    manager = MultiProcessManager()
    barrier = Barrier(clients_num)

    for i in range(clients_num):
        manager.add_proc(process, (barrier, f"article_{i}"))

    manager.run_all()
    print(manager.data)
    print(f"Avg. JCT {mean(list(manager.data.values())):.2f} (s)", flush=True)


if __name__ == "__main__":
    warmup()

    arg = int(sys.argv[1])

    main(arg)
