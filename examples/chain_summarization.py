# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from parrot import P

vm = P.VirtualMachine(
    core_http_addr="http://localhost:9000",
    mode="debug",
)


first_sum = vm.import_function("chain_summarize_first", "codelib.app.summarization")
refine = vm.import_function("chain_summarize_refine", "codelib.app.summarization")


def main():
    # Load docs
    docs_path = "data/state_of_the_union.txt"
    docs = open(docs_path, "r").read().split("\n\n")

    # Split into chunks and map
    chunk_size = 2000
    word_limit = 100

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
