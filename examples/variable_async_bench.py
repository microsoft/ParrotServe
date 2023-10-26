# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# Example from: https://python.langchain.com/docs/modules/chains/foundational/sequential_chains
# In this example, we create an automatic social media post writer, with
# the help of a "playwriter" and a "critic".


import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="debug",
)


@P.function(formatter=P.allowing_newline)
def func1(
    a: P.Output(max_gen_length=200, ignore_tokenizer_eos=True),
    b: P.Output(max_gen_length=200, ignore_tokenizer_eos=True),
):
    """This is a test function func1:

    First part:
    {{a}}

    Second part:
    {{b}}
    """


@P.function(formatter=P.allowing_newline)
def func2(
    a: P.Input,
    c: P.Output(max_gen_length=200, ignore_tokenizer_eos=True),
):
    """This is a test function func2:

    This is the prompt:
    {{a}}

    This is the output:
    {{c}}
    """


async def main():
    a, b = func1()
    b.get()
    c = func2(a)
    print(c.get())


vm.run(main(), timeit=True)
