# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This workload is created to show the acceleration of pipeling.


from parrot import env, P
import logging

# Disable the logging
logging.disable(logging.DEBUG)
logging.disable(logging.INFO)

env.register_tokenizer("hf-internal-testing/llama-tokenizer")
env.register_engine(
    "vicuna_7b_v1.3_local",
    host="localhost",
    port=8888,
    tokenizer="hf-internal-testing/llama-tokenizer",
)


@P.function()
def test_func(
    input_1: P.Input,
    input_2: P.Input,
    input_3: P.Input,
    output_1: P.Output,
    output_2: P.Output,
    output_3: P.Output,
):
    """This is a test function. It takes three inputs and produces three outputs.

    Input 1: {{input_1}}

    This text is intended to fill the middle part:
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ

    Input 2: {{input_2}}

    This text is intended to fill the middle part: ABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ

    Input 3: {{input_3}}

    This text is intended to fill the middle part: ABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ

    Output 1: {{output_1}}

    This text is intended to fill the middle part: ABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ

    Output 2: {{output_2}}

    This text is intended to fill the middle part: ABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ
    ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ

    Output 3: {{output_3}}
    """


async def main():
    start_funcs = 8
    layers = 6
    nodes = [
        [
            [P.placeholder(), P.placeholder(), P.placeholder()]
            for i in range(start_funcs)
        ]
        for j in range(layers)
    ]

    for i in range(start_funcs):
        for j in range(layers - 1):
            test_func(
                nodes[j][i][0],
                nodes[j][i][1],
                nodes[j][i][2],
                nodes[j + 1][i][0],
                nodes[j + 1][i][1],
                nodes[j + 1][i][2],
            )

    for i in range(start_funcs):
        nodes[0][i][0].assign("This is input 1")
        nodes[0][i][1].assign("This is input 2")
        nodes[0][i][2].assign("This is input 3")

    for i in range(start_funcs):
        print(
            await nodes[-1][i][0].get(),
            await nodes[-1][i][1].get(),
            await nodes[-1][i][2].get(),
        )


env.parrot_run_aysnc(main(), timeit=True)
