# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This notebook is a Tutorial for Parrot frontend syntax.
# We start from let the functions return "Hello World!" as an example.
# a.k.a The "Hello World!" semantic program!

import parrot as P

# We need to start a VM first before defining any functions, so that the
# functions can be registered to the environment.
# Also you can use `vm.import_function` to import functions from other modules.
vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="debug",
)


# Now we can start to define a "Parrot function".
# The magical thing is that, the function is "defined" by the
# docstring! (in a natural language way)
# The function will be automatically be registered to the environment


# We define a function called "print", which takes a string as input, and print it out.
# Different from traditional programming languages, we need some prompts to enforce
# LLMs to print the exact the same string we want.


@P.function()
def llm_print(string: P.Input, output: P.Output()):
    """You are a repeater. Given a string, it is your job to print it out.
    User input: {{string}}
    Your output: {{output}}"""


# Then we can start to define the main function.
def main():
    output = llm_print("Hello World!")  # print by the semantic code
    print(output.get())  # print by the native code


# Just run it. If your backend is intelligent enough, you will see the output is
# exactly "Hello World!".
vm.run(main)
