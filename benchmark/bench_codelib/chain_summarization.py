# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# Functions in this file are used for immitating the workload of chain summarization.

# Reference of prompt length: https://python.langchain.com/docs/use_cases/summarization

import parrot as P

fake_long_document_chunk = "Test " * 670  # len=670 for each chunk
refine_instruction = "Test " * 10


def chain_sum_test_order1(
    previous_document: P.Input,
    refined_document: P.Output(
        P.SamplingConfig(ignore_tokenizer_eos=True, max_gen_length=20)
    ),  # 20 Gen
):
    pass


chain_sum_test_body_order1 = (
    "{{previous_document}}"
    + f"{fake_long_document_chunk}"
    + f"{refine_instruction}"
    + "{{refined_document}}"
)

chain_sum_test_order1.__doc__ = chain_sum_test_body_order1
chain_sum_test_order1 = P.function(cache_prefix=False)(chain_sum_test_order1)

# print(chain_sum_test_body)
