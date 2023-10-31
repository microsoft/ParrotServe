# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="debug",
)


@P.function(formatter=P.allowing_newline)
def first_sum(
    doc: P.Input,
    summary: P.Output(temperature=0.5, max_gen_length=10, ignore_tokenizer_eos=True),
    word_limit: int,
):
    """The following is a piece of a document:
    {{doc}}
    Based on this piece of docs, please summarize the main content of this piece of docs as short as possible.
    The number of words should not exceed {{word_limit}}.
    Helpful Answer:
    {{summary}}
    """


@P.function(formatter=P.allowing_newline)
def refine(
    new_text: P.Input,
    previous_sum: P.Input,
    next_sum: P.Output(temperature=0.7, max_gen_length=10, ignore_tokenizer_eos=True),
    word_limit: int,
):
    """Your job is to produce a final summary

    We have the opportunity to refine the existing summary (only if needed) with some more context below.
    ------------
    {{new_text}}
    ------------
    Given the new context, refine the original summary in English.

    We have provided an existing summary up to a certain point: {{previous_sum}}

    If the context isn't useful, return the original summary.
    The number of words should not exceed {{word_limit}}.

    Helpful Answer:
    {{next_sum}}
    """


def main():
    # Load docs
    docs_path = "data/state_of_the_union.txt"
    docs = open(docs_path, "r").read().split("\n\n")

    # Split into chunks and map
    chunk_size = 2000
    word_limit = 10

    cur_chunk = ""
    i = 0

    for chunk in docs:
        cur_chunk += chunk
        if len(cur_chunk) > chunk_size:
            print("Created chunk of size", len(cur_chunk))
            if i == 0:
                previous_sum = first_sum(cur_chunk, word_limit)
            else:
                previous_sum = refine(
                    new_text=cur_chunk,
                    previous_sum=previous_sum,
                    word_limit=word_limit,
                )
            # _test = previous_sum.get()  # this is a hack to ban the variable-async
            i += 1

            cur_chunk = ""
    print("Total number of chunks:", i)

    final_sumary = previous_sum.get()
    print("The following is the final summary of the document:\n", final_sumary)


# main()

vm.run(main, timeit=True)
