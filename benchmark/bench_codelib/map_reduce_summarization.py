# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# Functions in this file are used for immitating the workload of chain summarization.

# Reference of prompt length: https://python.langchain.com/docs/use_cases/summarization

import parrot as P

map_instruction_1 = "Test " * 10
map_instruction_2 = "Test " * 10
reduce_instruction_1 = "Test " * 10
reduce_instruction_2 = "Test " * 10


# NOTE(chaofan): For baseline, we use the same upperbound as the main.
def map_sum_test_baseline(
    doc_chunk: P.Input,
    doc_sum: P.Output(
        P.SamplingConfig(ignore_tokenizer_eos=True, max_gen_length=50),
        P.DispatchAnnotation(requests_num_upperbound=2),
    ),  # 20 Gen
):
    pass


# NOTE(chaofan): We annotate map stage a large job upperbound, 
# and reduce stage a small job upperbound.
def map_sum_test_main(
    doc_chunk: P.Input,
    doc_sum: P.Output(
        P.SamplingConfig(ignore_tokenizer_eos=True, max_gen_length=50),
        P.DispatchAnnotation(requests_num_upperbound=64),
    ),  # 20 Gen
):
    pass


map_sum_test_body = (
    f"{map_instruction_1}"
    + "{{doc_chunk}}"
    + f"{map_instruction_2}"
    + "{{doc_sum}}"
)

map_sum_test_baseline.__doc__ = map_sum_test_body
map_sum_test_baseline = P.function(cache_prefix=False)(map_sum_test_baseline)
map_sum_test_main.__doc__ = map_sum_test_body
map_sum_test_main = P.function(cache_prefix=False)(map_sum_test_main)


# We unrolling the function for now.
chunk_num = 15
def reduce_sum_test(
    chunk_sum_1: P.Input,
    chunk_sum_2: P.Input,
    chunk_sum_3: P.Input,
    chunk_sum_4: P.Input,
    chunk_sum_5: P.Input,
    chunk_sum_6: P.Input,
    chunk_sum_7: P.Input,
    chunk_sum_8: P.Input,
    chunk_sum_9: P.Input,
    chunk_sum_10: P.Input,
    chunk_sum_11: P.Input,
    chunk_sum_12: P.Input,
    chunk_sum_13: P.Input,
    chunk_sum_14: P.Input,
    chunk_sum_15: P.Input,
    output: P.Output(
        P.SamplingConfig(ignore_tokenizer_eos=True, max_gen_length=50)
    ),  # 50 Gen
):
    pass


reduce_sum_test_body = (
    f"{reduce_instruction_1}" + \
    "".join(["{{chunk_sum_" + str(i+1) + "}}" for i in range(chunk_num)]) + \
    f"{reduce_instruction_2}" + "{{output}}"
)

reduce_sum_test.__doc__ = reduce_sum_test_body
reduce_sum_test = P.function(cache_prefix=False)(reduce_sum_test)