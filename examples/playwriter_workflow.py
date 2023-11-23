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
def write_synopsis(
    title: P.Input,
    era: P.Input,
    synopsis: P.Output(P.SamplingConfig(max_gen_length=200, ignore_tokenizer_eos=True)),
):
    """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {{title}}
    Era: {{era}}
    Playwright: This is a synopsis for the above play: {{synopsis}}"""


@P.function(formatter=P.allowing_newline)
def write_review(
    synopsis: P.Input,
    review: P.Output(P.SamplingConfig(max_gen_length=200, ignore_tokenizer_eos=True)),
):
    """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {{synopsis}}
    Review from a New York Times play critic of the above play: {{review}}"""


@P.function(formatter=P.allowing_newline)
def write_post(
    time: P.Input,
    location: P.Input,
    synopsis: P.Input,
    review: P.Input,
    post: P.Output(P.SamplingConfig(max_gen_length=200, ignore_tokenizer_eos=True)),
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

    review.get()

    post = write_post(
        time="December 25th, 8pm PST",
        location="Theater in the Park",
        synopsis=synopsis,
        review=review,
    )

    print("---------- Play Synopsis ----------")
    print(await synopsis.aget())

    print("---------- Review ----------")
    print(await review.aget())

    print("---------- Social Media Post ----------")
    print(await post.aget())


vm.run(main(), timeit=True)
