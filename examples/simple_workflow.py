# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# Example from: https://python.langchain.com/docs/modules/chains/foundational/sequential_chains
# In this example, we create an automatic social media post writer, with
# the help of a "playwriter" and a "critic".


import parrot as P

vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.init()


@P.function()
def write_synopsis(
    title: P.Input,
    era: P.Input,
    synopsis: P.Output,
):
    """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {{title}}
    Era: {{era}}
    Playwright: This is a synopsis for the above play: {{synopsis}}"""


@P.function()
def write_review(
    synopsis: P.Input,
    review: P.Output,
):
    """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {{synopsis}}
    Review from a New York Times play critic of the above play: {{review}}"""


@P.function()
def write_post(
    time: P.Input,
    location: P.Input,
    synopsis: P.Input,
    review: P.Input,
    post: P.Output,
):
    """You are a social media manager for a theater company. Given the title of play, the era it is set in, the date, time and location, the synopsis of the play, and the review of the play, it is your job to write a social media post for that play.

    Here is some context about the time and location of the play:
    Date and Time: {{time}}
    Location: {{location}}

    Play Synopsis:
    {{synopsis}}
    Review from a New York Times play critic of the above play:
    {{review}}

    Social Media Post: {{post}}
    """


async def main():
    synopsis = write_synopsis(
        title="Tragedy at sunset on the beach", era="Victorian England"
    )
    review = write_review(synopsis)
    post = write_post(
        time="December 25th, 8pm PST",
        location="Theater in the Park",
        synopsis=synopsis,
        review=review,
    )

    print("---------- Play Synopsis ----------")
    print(await synopsis.get())

    print("---------- Review ----------")
    print(await review.get())

    print("---------- Social Media Post ----------")
    print(await post.get())


vm.run(main(), timeit=True)
