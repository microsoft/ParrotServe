# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This application is a generator of jokes and their explanations.
# It can generate a batch of jokes at a time!


import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="debug",
)


joke_generator = vm.import_function("tell_me_a_joke", "codelib.app.common")


def main():
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
        joke, explanation = joke_generator(topics[i], topic2s[i])
        jokes.append(joke)
        explanations.append(explanation)

    for i in range(len(topics)):
        joke_str = jokes[i].get()
        print(f"---------- Round {i}: The following is the joke ---------- ")
        print(joke_str)
        print(
            f"---------- If you don't get it, the following is the explanation ---------- "
        )
        print(explanations[i].get())


vm.run(main)
