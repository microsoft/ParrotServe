# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from parrot import env, P
import logging

# Disable the logging
logging.disable(logging.DEBUG)
logging.disable(logging.INFO)

env.register_tokenizer("hf-internal-testing/llama-tokenizer")
env.register_engine(
    "vicuna_13b_v1.3_local",
    host="localhost",
    port=8888,
    tokenizer="hf-internal-testing/llama-tokenizer",
)


@P.function()
def qa(
    question: P.Input,
    answer: P.Output,
):
    """You are a helpful assistant who can answer questions. For each question, you
    should answer it correctly and concisely. And try to make the answer as short as possible (Ideally,
    just one or two words).

    The question is: {{question}}.

    The answer is: {{answer}}.
    """


async def main():
    while True:
        question_str = input("Please input your question: ")
        question = P.placeholder()
        answer = P.placeholder()
        qa(question, answer)
        question.assign(question_str)
        answer_str = await answer.get()
        print(f"---------- The following is the answer ---------- ")
        print(answer_str)


env.parrot_run_aysnc(main())
