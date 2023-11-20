# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# Functions in this file are used for immitating the ReAct workloads,
# e.g. Chatbots, Agent Role-Play, etc.

import parrot as P


def normal_multi_stage_react_agent(
    stage0_input: P.Input,
    stage0_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage1_input: P.Input,
    stage1_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage2_input: P.Input,
    stage2_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage3_input: P.Input,
    stage3_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage4_input: P.Input,
    stage4_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage5_input: P.Input,
    stage5_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage6_input: P.Input,
    stage6_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage7_input: P.Input,
    stage7_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage8_input: P.Input,
    stage8_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
    stage9_input: P.Input,
    stage9_output: P.Output(ignore_tokenizer_eos=True, max_gen_length=40),
):
    """Long system prompt:

    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    {{stage0_input}}
    {{stage0_output}}

    {{stage1_input}}
    {{stage1_output}}

    {{stage2_input}}
    {{stage2_output}}

    {{stage3_input}}
    {{stage3_output}}

    {{stage4_input}}
    {{stage4_output}}

    {{stage5_input}}
    {{stage5_output}}

    {{stage6_input}}
    {{stage6_output}}

    {{stage7_input}}
    {{stage7_output}}

    {{stage8_input}}
    {{stage8_output}}

    {{stage9_input}}
    {{stage9_output}}
    """
