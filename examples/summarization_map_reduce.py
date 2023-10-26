# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="debug",
)


@P.function(formatter=P.allowing_newline)
def map(
    doc_pieces: P.Input,
    summary: P.Output(temperature=0.5, max_gen_length=50),
):
    """The following is a piece of a document:
    {{doc_pieces}}
    Based on this piece of docs, please summarize the main content of this piece of docs as short as possible.
    Helpful Answer:
    {{summary}}
    """


@P.function(formatter=P.allowing_newline)
def reduce(
    doc_summaries: P.Input,
    final_summary: P.Output(temperature=0.7, max_gen_length=200),
):
    """The following is set of summaries:

    {{doc_summaries}}

    Take these and distill it into a final, consolidated summary of the main themes as short as possible..
    Helpful Answer:
    {{final_summary}}
    """


def main():
    # Load docs
    docs_path = "data/state_of_the_union.txt"
    docs = open(docs_path, "r").read().split("\n\n")

    # Split into chunks and map
    chunk_size = 1200
    cur_chunk = ""
    summaries_list = []
    for chunk in docs:
        cur_chunk += chunk
        if len(cur_chunk) > chunk_size:
            print("Created chunk of size", len(cur_chunk))
            summaries_list.append(map(cur_chunk))
            cur_chunk = ""
    print("Total number of chunks:", len(summaries_list))

    # Reduce
    summaries = ""
    for i, summary in enumerate(summaries_list):
        summaries += summary.get()
        if i != len(summaries_list) - 1:
            summaries += "\n\n"

    final_summary = reduce(summaries)
    print("The following is the final summary of the document:\n", final_summary.get())


# main()

vm.run(main)
