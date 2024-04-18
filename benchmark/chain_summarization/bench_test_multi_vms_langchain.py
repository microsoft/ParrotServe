# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time
from numpy import mean
from multiprocessing import Barrier
from parrot.testing.multiproc_manager import MultiProcessManager

module = importlib.import_module(
    "benchmark.workloads.test_examples.chain_summarization"
)
fake_long_document_chunk = getattr(module, "fake_long_document_chunk")

chunk_num = 20

full_document = (fake_long_document_chunk + "\n\n") * chunk_num

with open("test.txt", "w") as f:
    f.write(full_document)


def process(barrier: Barrier):
    output_len = 50

    ### Langchain part

    from langchain.chains.summarize import load_summarize_chain
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=output_len)
    chain = load_summarize_chain(llm, chain_type="refine")

    loader = TextLoader("test.txt")
    docs = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=650,
        chunk_overlap=0,
    )
    split_docs = text_splitter.split_documents(docs)

    # for i, doc in enumerate(split_docs):
    #     print(i, len(doc.page_content.split(" ")))

    barrier.wait()
    st = time.perf_counter_ns()
    chain.run(split_docs)
    ed = time.perf_counter_ns()
    return (ed - st) / 1e9


clients_num = 8
manager = MultiProcessManager()
barrier = Barrier(clients_num)

for _ in range(clients_num):
    manager.add_proc(process, (barrier,))

manager.run_all()
print(manager.data)
print(f"Avg. JCT {mean(list(manager.data.values())):.2f} (s)")
