# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import asyncio
import time


### Langchain part

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=50)

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


async def amap(doc: str):
    resp = await map_chain.arun(text=doc)
    return resp


async def run(split_docs):
    coros = []
    for doc in split_docs:
        coros.append(amap(doc=doc.page_content))

    st = time.perf_counter_ns()
    docs = await asyncio.gather(*coros)
    docs = "\n".join(docs)
    # print(docs)
    resp = await reduce_chain.arun(docs=docs)
    ed = time.perf_counter_ns()
    latency = (ed - st) / 1e9
    print(f"Time: {latency:.4f}", flush=True)


def main(file_name: str, chunk_size: int, output_len: int):
    print(
        f"file_name: {file_name}, chunk_size: {chunk_size}, output_len: {output_len}",
        flush=True,
    )

    docs = prepare_docs(file_name, chunk_size)
    asyncio.run(run(docs))

    time.sleep(3)


def warmup():
    map_chain.run("This is a test")


if __name__ == "__main__":
    warmup()

    print("warmup done", flush=True)

    for i in range(10):
        for ol in [25, 50, 75, 100]:
            main(f"article_{i}", 1024, ol)
