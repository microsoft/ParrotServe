# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time

module = importlib.import_module(f"benchmark.bench_codelib.map_reduce_summarization")
chunk_num = getattr(module, "chunk_num_30")
map_document_chunk = "Test " * 1000  # len=1000 for each chunk

full_document = (map_document_chunk + "\n\n") * chunk_num

with open("test.txt", "w") as f:
    f.write(full_document)


### Langchain part

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain = load_summarize_chain(llm, chain_type="map_reduce")

loader = TextLoader("test.txt")

docs = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024,
    chunk_overlap=0,
)
split_docs = text_splitter.split_documents(docs)

for i, doc in enumerate(split_docs):
    print(i, len(doc.page_content.split(" ")))

for _ in range(10):
    st = time.perf_counter_ns()
    chain.run(split_docs)
    ed = time.perf_counter_ns()
    with open("langchain_stdout.log", "a+") as f:
        print(f"Time: {(ed - st) / 1e9} s", file=f, flush=True)
    time.sleep(3)
