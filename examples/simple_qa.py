# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P
import aioconsole  # We use aioconsole to read input asynchronously
import logging

vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.init()


@P.function(formatter=P.allowing_newline)
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
        question = aioconsole.ainput("Your question: ")
        answer = qa(question)
        print("Answer: ", await answer.get())


vm.run(main())
