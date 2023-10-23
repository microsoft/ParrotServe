# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="release",
)


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


def main():
    while True:
        question = input("Your question: ")
        answer = qa(question)
        print("Answer: ", answer.get())


vm.run(main)
