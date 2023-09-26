# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.init()


@P.function()
def tell_me_a_joke(
    topic: P.Input,
    keyword: P.Input,
    joke: P.Output,
    explanation: P.Output,
):
    """Tell me a short joke about {{topic}}. The joke must contains the following
    keywords: {{keyword}}. The following is the joke: {{joke}}. And giving a
    short explanation to show that why it is funny. The following is the explanation
    for the joke above: {{explanation}}."""


async def main():
    topics = [
        "a programmer",
        "a mathematician",
        "a physicist",
    ]
    keywords = [
        "bug",
        "iPhone",
        "cat",
    ]

    for i in range(3):
        joke, explanation = tell_me_a_joke(topics[i], keywords[i])
        joke_str = await joke.get()
        print(f"---------- Round {i}: The following is the joke ---------- ")
        print(joke_str)
        print(
            f"---------- If you don't get it, the following is the explanation ---------- "
        )
        print(await explanation.get())


vm.run(main())
