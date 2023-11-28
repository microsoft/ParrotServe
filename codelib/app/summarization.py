# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This module contains functions for text summarization.

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


### ChainSummarization Functions Start


@P.semantic_function(formatter=P.allowing_newline)
def chain_summarize_first(
    doc: P.Input,
    summary: P.Output(P.SamplingConfig(temperature=0.5, max_gen_length=200)),
    word_limit: int,
):
    """The following is a piece of a document:
    {{doc}}
    Based on this piece of docs, please summarize the main content of this piece of docs as short as possible.
    The number of words should not exceed {{word_limit}}.
    Helpful Answer:
    {{summary}}
    """


@P.semantic_function(formatter=P.allowing_newline)
def chain_summarize_refine(
    new_text: P.Input,
    previous_sum: P.Input,
    next_sum: P.Output(P.SamplingConfig(temperature=0.7, max_gen_length=200)),
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


### ChainSummarization Functions End
