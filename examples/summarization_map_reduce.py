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
    chunk1: P.Input,
    chunk2: P.Input,
    chunk3: P.Input,
    chunk4: P.Input,
    chunk5: P.Input,
    chunk6: P.Input,
    chunk7: P.Input,
    chunk8: P.Input,
    chunk9: P.Input,
    chunk10: P.Input,
    chunk11: P.Input,
    chunk12: P.Input,
    chunk13: P.Input,
    chunk14: P.Input,
    chunk15: P.Input,
    chunk16: P.Input,
    final_summary: P.Output(temperature=0.7, max_gen_length=200),
):
    """The following is set of summaries:

    {{chunk1}}
    {{chunk2}}
    {{chunk3}}
    {{chunk4}}
    {{chunk5}}
    {{chunk6}}
    {{chunk7}}
    {{chunk8}}
    {{chunk9}}
    {{chunk10}}
    {{chunk11}}
    {{chunk12}}
    {{chunk13}}
    {{chunk14}}
    {{chunk15}}
    {{chunk16}}

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

    if len(summaries_list) > 16:
        raise RuntimeError("Too many chunks")

    # Reduce
    final_summary = reduce(*summaries_list)
    print("The following is the final summary of the document:\n", final_summary.get())


# main()

vm.run(main)
