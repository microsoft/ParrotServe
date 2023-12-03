# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import importlib
import time


chunk_size = 1024
file_name = "article_5"


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

prompt_template = """Write an one-sentence summary (AS SHORT AS POSSIBLE) of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce an one-sentence summary (AS SHORT AS POSSIBLE) for a long document.\n"
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

output_lens = []
with open(
    f"arxiv-sampled/{file_name}-chain-outputlen.txt", encoding="utf-8", mode="w"
) as f:
    for i, step in enumerate(steps):
        output_len = len(tokenizer.encode(step, add_special_tokens=False))
        output_lens.append(output_len)
        print(
            f"Step {i}: Output Len={output_len}",
            file=f,
            flush=True,
        )

    print("Average output length:", sum(output_lens) / len(output_lens), file=f)
