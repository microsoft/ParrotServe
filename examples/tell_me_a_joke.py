# Copyright (c) 2023 by Microsoft Corporation.

# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from parrot import env, P
import logging

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
def tell_me_a_joke(
    topic: P.Input,
    keyword: P.Input,
    joke: P.Output,
    explanation: P.Output,
):
    """Tell me a joke about {{topic}}. The joke must contains the following
    keywords: {{keyword}}. The following is the joke: {{joke}}. And giving a
    short explanation to show that why it is funny. The following is the explanation
    for the joke above: {{explanation}}."""


async def main():
    topic = P.placeholder()
    keyword = P.placeholder()
    joke = P.placeholder()
    explanation = P.placeholder()

    tell_me_a_joke(topic, keyword, joke, explanation)

    topic.assign("a programmer")
    keyword.assign("bug")
    joke_str = await joke.get()
    print("\n\n ---------- The following is the joke ---------- ")
    print(joke_str)
    print(
        "\n\n ---------- If you don't get it, the following is the explanation ---------- "
    )
    print(await explanation.get())


env.parrot_run_aysnc(main())
