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
    ),
):
    pass


# NOTE(chaofan): We annotate map stage a large job upperbound,
# and reduce stage a small job upperbound.
def map_sum_test_main(
    doc_chunk: P.Input,
    doc_sum: P.Output(
        P.SamplingConfig(ignore_tokenizer_eos=True, max_gen_length=50),
        P.DispatchAnnotation(requests_num_upperbound=32),
    ),
):
    pass


map_sum_test_body = (
    f"{map_instruction_1}" + "{{doc_chunk}}" + f"{map_instruction_2}" + "{{doc_sum}}"
)

map_sum_test_baseline.__doc__ = map_sum_test_body
map_sum_test_baseline = P.function(cache_prefix=False)(map_sum_test_baseline)
map_sum_test_main.__doc__ = map_sum_test_body
map_sum_test_main = P.function(cache_prefix=False)(map_sum_test_main)


# Setting 1

chunk_num_15 = 15


# We unrolling the function for now.
def reduce_sum_test_15(
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
    output: P.Output(P.SamplingConfig(ignore_tokenizer_eos=True, max_gen_length=50)),
):
    pass


reduce_sum_test_body_15 = (
    f"{reduce_instruction_1}"
    + "".join(["{{chunk_sum_" + str(i + 1) + "}}" for i in range(chunk_num_15)])
    + f"{reduce_instruction_2}"
    + "{{output}}"
)

reduce_sum_test_15.__doc__ = reduce_sum_test_body_15
reduce_sum_test_15 = P.function(cache_prefix=False)(reduce_sum_test_15)


# Setting 2

chunk_num_30 = 30


# We unrolling the function for now.
def reduce_sum_test_30(
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
    chunk_sum_16: P.Input,
    chunk_sum_17: P.Input,
    chunk_sum_18: P.Input,
    chunk_sum_19: P.Input,
    chunk_sum_20: P.Input,
    chunk_sum_21: P.Input,
    chunk_sum_22: P.Input,
    chunk_sum_23: P.Input,
    chunk_sum_24: P.Input,
    chunk_sum_25: P.Input,
    chunk_sum_26: P.Input,
    chunk_sum_27: P.Input,
    chunk_sum_28: P.Input,
    chunk_sum_29: P.Input,
    chunk_sum_30: P.Input,
    output: P.Output(P.SamplingConfig(ignore_tokenizer_eos=True, max_gen_length=50)),
):
    pass


reduce_sum_test_body_30 = (
    f"{reduce_instruction_1}"
    + "".join(["{{chunk_sum_" + str(i + 1) + "}}" for i in range(chunk_num_30)])
    + f"{reduce_instruction_2}"
    + "{{output}}"
)

reduce_sum_test_30.__doc__ = reduce_sum_test_body_30
reduce_sum_test_30 = P.function(cache_prefix=False)(reduce_sum_test_30)
