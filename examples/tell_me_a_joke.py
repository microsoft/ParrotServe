# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.init()


@P.function(conversation_template=P.vicuna_template)
def tell_me_a_joke(
    topic: P.Input,
    topic2: P.Input,
    joke: P.Output,
    explanation: P.Output(temperature=0.5),
):
    """Tell the me a joke about {{topic}} and {{topic2}}. {{joke}}.
    Good, then giving a short explanation to show that why it is funny.
    The explanation should be short, concise and clear. {{explanation}}.
    """


async def main():
    topics = [
        "student",
        "machine learning",
        "human being",
        "a programmer",
        "a mathematician",
        "a physicist",
    ]
    topic2s = [
        "homework",
        "monkey",
        "robot",
        "bug",
        "iPhone",
        "cat",
    ]
    jokes = []
    explanations = []

    for i in range(len(topics)):
        joke, explanation = tell_me_a_joke(topics[i], topic2s[i])
        jokes.append(joke)
        explanations.append(explanation)

    for i in range(len(topics)):
        joke_str = await jokes[i].get()
        print(f"---------- Round {i}: The following is the joke ---------- ")
        print(joke_str)
        print(
            f"---------- If you don't get it, the following is the explanation ---------- "
        )
        print(await explanations[i].get())


vm.run(main())
