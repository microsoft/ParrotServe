# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import asyncio
import time
import sys


### Langchain part

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer


def prepare_chains(output_len: int):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=output_len)

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
    return map_chain, reduce_chain


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
    # print("chunk_num: ", len(split_docs), flush=True)
    # for doc in split_docs:
    #     content = doc.page_content
    #     token_ids = tokenizer.tokenize(content)
    #     print(len(token_ids), flush=True)
    return split_docs


async def amap(map_chain, doc: str):
    resp = await map_chain.arun(text=doc)
    # print("Received resp", flush=True)
    return resp


async def run(map_chain, reduce_chain, split_docs):
    coros = []
    for doc in split_docs:
        coros.append(amap(map_chain=map_chain, doc=doc.page_content))

    st = time.perf_counter_ns()
    try:
        docs = await asyncio.gather(*coros, return_exceptions=True)
    except Exception as e:
        pass
    docs = "\n".join(docs)
    resp = reduce_chain.run(docs=docs)
    ed = time.perf_counter_ns()
    latency = (ed - st) / 1e9
    print(f"Time: {latency:.4f}", flush=True)


def main(file_name: str, chunk_size: int, output_len: int):
    print(
        f"file_name: {file_name}, chunk_size: {chunk_size}, output_len: {output_len}",
        flush=True,
    )

    map_chain, reduce_chain = prepare_chains(output_len)

    docs = prepare_docs(file_name, chunk_size)

    asyncio.run(run(map_chain, reduce_chain, docs))


def warmup():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=5)
    llm.invoke("hello")


if __name__ == "__main__":
    warmup()

    # main("article_0", 1024, 75)
    # time.sleep(3)
    # main("article_0", 1024, 100)
    # 0 / 0

    arg = sys.argv[1]
    article_id = int(sys.argv[2])

    if arg == "test":
        main("article_0", 1024, 100)
    elif arg == "exp1":
        for ol in [25, 50, 75, 100]:
            main(f"article_{article_id}", 1024, ol)
    elif arg == "exp2":
        for cs in [512, 1024, 1536, 2048]:
            main(f"article_{article_id}", cs, 50)
