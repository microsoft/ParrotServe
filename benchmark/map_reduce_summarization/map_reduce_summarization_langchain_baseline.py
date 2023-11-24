# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time

lib_path: str = "semantic_code_lib"
module_path: str = "bench.map_reduce_summarization"

module = importlib.import_module(f"{lib_path}.{module_path}")
chunk_num = getattr(module, "chunk_num")
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
time.sleep(3) # take 3 second to load the docs from web, suppose

docs = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024,
    chunk_overlap=0,
)
split_docs = text_splitter.split_documents(docs)

for i, doc in enumerate(split_docs):
    print(i, len(doc.page_content.split(" ")))

st = time.perf_counter_ns()
chain.run(split_docs)
ed = time.perf_counter_ns()
print(f"Time: {(ed - st) / 1e9} s")
