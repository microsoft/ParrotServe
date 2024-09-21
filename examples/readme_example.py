# Copyright (c) 2024 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from parrot import P

vm = P.VirtualMachine(
    core_http_addr="http://localhost:9000",
    mode="debug",
)


@P.semantic_function()
def tell_me_a_joke(topic: P.Input, joke: P.Output):
    """Tell the me a joke about {{topic}}: {{joke}}."""


@P.native_function()
def format_joke(joke: P.Input, formatted_joke: P.Output):
    ret = (
        "Here is the joke for you\n---\n" + joke
    )  # Directly use string built-in methods
    formatted_joke.set(ret)  # Use `set` to assign value to output


def main():  # Orchestrator function
    joke = tell_me_a_joke(topic="chicken")
    joke1 = format_joke(joke)
    joke_str = joke1.get()
    print(joke_str)


vm.run(main)
