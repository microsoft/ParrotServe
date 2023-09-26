# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.init()


@P.function()
def tell_me_a_joke(
    topic: P.Input,
    topic2: P.Input,
    joke: P.Output,
    explanation: P.Output,
    word_limit: int,
):
    """Tell the me a joke about {{topic}} and {{topic2}}.
    The joke is limited to {{word_limit}} characters.
    Don't generate a joke that is too long.
    Only generate a single joke.
    Sure, here's a joke for you: {{joke}}. Good, and giving a
    short explanation to show that why it is funny.
    The explanation should be short, concise and clear.
    Sure, here's a short explanation for the joke above: {{explanation}}."""


async def main():
    topics = [
        "a programmer",
        "a mathematician",
        "a physicist",
    ]
    topic2s = [
        "bug",
        "iPhone",
        "cat",
    ]
    jokes = []
    explanations = []

    for i in range(3):
        joke, explanation = tell_me_a_joke(topics[i], topic2s[i], 100)
        jokes.append(joke)
        explanations.append(explanation)

    for i in range(3):
        joke_str = await jokes[i].get()
        print(f"---------- Round {i}: The following is the joke ---------- ")
        print(joke_str)
        print(
            f"---------- If you don't get it, the following is the explanation ---------- "
        )
        print(await explanations[i].get())


vm.run(main())
