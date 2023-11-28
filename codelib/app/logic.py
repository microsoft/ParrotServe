# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This module contains functions for logic, including inference, math, etc.

import parrot as P


### MapReduce Functions Start

# Reference: https://python.langchain.com/docs/use_cases/summarization


@P.semantic_function(formatter=P.allowing_newline)
def summarize_map(
    doc_pieces: P.Input,
    summary: P.Output(P.SamplingConfig(temperature=0.5, max_gen_length=50)),
):
    """The following is a piece of a document:
    {{doc_pieces}}
    Based on this piece of docs, please summarize the main content of this piece of docs as short as possible.
    Helpful Answer:
    {{summary}}
    """


@P.semantic_function(formatter=P.allowing_newline)
def summarize_reduce(
    doc_summaries: P.Input,
    final_summary: P.Output(P.SamplingConfig(temperature=0.7, max_gen_length=200)),
):
    """The following is set of summaries:

    {{doc_summaries}}

    Take these and distill it into a final, consolidated summary of the main themes as short as possible..
    Helpful Answer:
    {{final_summary}}
    """


### MapReduce Functions End
