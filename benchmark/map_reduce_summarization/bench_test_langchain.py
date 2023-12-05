# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import asyncio
import importlib
import time

module = importlib.import_module(
    f"benchmark.workloads.test_examples.map_reduce_summarization"
)
chunk_num = getattr(module, "chunk_num_15")
chunk_size = 1000
map_document_chunk = "Test " * chunk_size

full_document = (map_document_chunk + "\n\n") * chunk_num

with open("test.txt", "w") as f:
    f.write(full_document)


### Langchain part

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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


loader = TextLoader("test.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size,
    chunk_overlap=0,
)
split_docs = text_splitter.split_documents(docs)

for i, doc in enumerate(split_docs):
    print(i, len(doc.page_content.split(" ")))


async def amap(idx: int):
    resp = await map_chain.arun(text=split_docs[idx].page_content)
    return resp


async def main():
    coros = []
    for i in range(chunk_num):
        coros.append(amap(i))

    st = time.perf_counter_ns()
    docs = await asyncio.gather(*coros)
    docs = "\n".join(docs)
    # print(docs)
    resp = await reduce_chain.arun(docs=docs)
    ed = time.perf_counter_ns()
    with open("langchain_stdout.log", "a+") as f:
        print(f"Time: {(ed - st) / 1e9} s", file=f, flush=True)


for _ in range(10):
    asyncio.run(main())
