# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This application is a text summarization agent, which uses Map-Reduce strategy
# to handle long documents.

import asyncio
from parrot import P

vm = P.VirtualMachine(
    core_http_addr="http://localhost:9000",
    mode="debug",
)


map = vm.import_function("summarize_map", "codelib.app.summarization")
reduce = vm.import_function("summarize_reduce", "codelib.app.summarization")


async def main():
    # Load docs
    docs_path = "data/state_of_the_union.txt"
    docs = open(docs_path, "r").read().split("\n\n")

    # Split into chunks and map
    chunk_size = 1200
    cur_chunk = ""
    summaries_list = []
    coroutines = []
    for chunk in docs:
        cur_chunk += chunk
        if len(cur_chunk) > chunk_size:
            print("Created chunk of size", len(cur_chunk))
            future = map(cur_chunk)
            coroutines.append(future.aget())
            cur_chunk = ""
    print("Total number of chunks:", len(summaries_list))

    # Reduce
    result = await asyncio.gather(*coroutines)
    summaries = "\n".join(result)
    final_summary = reduce(summaries)
    print("The following is the final summary of the document:\n", final_summary.get())


# main()

vm.run(main)
