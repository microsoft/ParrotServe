# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time


chunk_size = 1024
file_name = "article_0"


### Langchain part

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer

llm = AzureChatOpenAI(temperature=0, model_name="gpt-4-32k")
loader = TextLoader(f"arxiv-sampled/{file_name}.txt")
docs = loader.load()

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=chunk_size,
    chunk_overlap=0,
    separator=" ",
)
split_docs = text_splitter.split_documents(docs)

for i, doc in enumerate(split_docs):
    print(i, len(tokenizer.encode(doc.page_content)))

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
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain({"input_documents": split_docs}, return_only_outputs=True)
steps = result["intermediate_steps"]

with open(
    f"arxiv-sampled/{file_name}-chain-outputlen.txt", encoding="utf-8", mode="w"
) as f:
    for i, step in enumerate(steps):
        print(
            f"Step {i}: Output Len={str(len(tokenizer.encode(step)))}",
            file=f,
            flush=True,
        )
