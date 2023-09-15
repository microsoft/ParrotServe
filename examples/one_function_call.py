# Copyright (c) 2023 by Microsoft Corporation.

# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from parrot import env, P

# We need to configure the environment before we can use it.
# Here we use the OPT model from facebook as an example.

# First, we need to register the tokenizer we want to use.
env.register_tokenizer("facebook/opt-13b")
# Then, we need to register the engine we want to use.
env.register_engine(
    "opt_13b_local",
    host="localhost",
    port=8888,
    tokenizer="facebook/opt-13b",
)


# Now we can start to define a "Parrot function".
# The magical thing is that, the function is "defined" by the
# docstring! (in a natural language way)
# The function will be automatically be registered to the environment


@P.function()
def tell_me_a_joke(
    topic: P.Input,
    keyword: P.Input,
    joke: P.Output,
    explanation: P.Output,
):
    """Tell me a joke about {{topic}}. The following is the joke: {{joke}}. And giving a
    short explanation to show that why it is funny. The following is the explanation
    for the joke above: {{explanation}}."""


# Now we can call the function we just defined.


async def main():
    # First we need some placeholders.
    topic = P.placeholder()
    joke = P.placeholder()
    explanation = P.placeholder()

    topic.assign("parrots")

    # Then we can call the function.
    tell_me_a_joke(
        topic=topic,
        joke=joke,
        explanation=explanation,
    )

    print(f"Request: Tell me a joke about {await topic.get()}.")
    joke_text = await joke.get()
    print(f"The joke: {joke_text}")
    explanation_text = await explanation.get()
    print(f"The explanation: {explanation_text}")


env.parrot_run_aysnc(main())
