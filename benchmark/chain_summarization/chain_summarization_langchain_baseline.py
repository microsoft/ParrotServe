# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time

module = importlib.import_module(f"benchmark.bench_codelib.chain_summarization")
fake_long_document_chunk = getattr(module, "fake_long_document_chunk")

chunk_num = 20

full_document = (fake_long_document_chunk + "\n\n") * chunk_num

with open("test.txt", "w") as f:
    f.write(full_document)


### Langchain part

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain = load_summarize_chain(llm, chain_type="refine")

loader = TextLoader("test.txt")
docs = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=670,
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
