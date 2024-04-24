# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import sys
import time


### Langchain part

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer


def main(file_name: str, chunk_size: int, output_len: int):
    print(
        f"file_name: {file_name}, chunk_size: {chunk_size}, output_len: {output_len}",
        flush=True,
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=output_len)
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

    # for i, doc in enumerate(split_docs):
    #     print(i, len(tokenizer.encode(doc.page_content)))

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "!!!IMPORTANT!!! Never let your summary exceeds 50 words.\n"
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

    for _ in range(1):
        st = time.perf_counter_ns()
        result = run_chain()
        ed = time.perf_counter_ns()

        print(f"Time: {(ed - st) / 1e9:.4f}s", flush=True)

        time.sleep(3)


def warmup():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=5)
    llm.invoke("hello")


if __name__ == "__main__":
    warmup()

    arg = sys.argv[1]
    article_id = int(sys.argv[2])

    if arg == "test":
        main("article_0", 1024, 100)
    elif arg == "exp1":
        if article_id == -1:
            for i in range(10):
                for ol in [25, 50, 75, 100]:
                    main(f"article_{i}", 1024, ol)
        else:
            for ol in [25, 50, 75, 100]:
                main(f"article_{article_id}", 1024, ol)
    elif arg == "exp2":
        if article_id == -1:
            for i in range(10):
                for cs in [512, 1024, 1536, 2048]:
                        main(f"article_{i}", cs, 50)
        else:
            for cs in [512, 1024, 1536, 2048]:
                main(f"article_{article_id}", cs, 50)
