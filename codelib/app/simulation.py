# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This module contains functions for making LLM to act as a simulator for a real-world
# software, like Linux terminal, SQL, Web server, etc.

import parrot as P


# Reference: https://github.com/f/awesome-chatgpt-prompts
# Act as a Linux Terminal


@P.function()
def linux_terminal(
    command: P.Input,
    output: P.Output(P.SamplingConfig(temperature=0.5)),
):
    """I want you to act as a linux terminal.
    I will type commands and you will reply with what the terminal should show.
    I want you to only reply with the terminal output inside one unique code block, and nothing else.
    do not write explanations. do not type commands unless I instruct you to do so.
    When I need to tell you something in English,
    I will do so by putting text inside curly brackets {like this}.
    My first command is {{command}}.
    {{output}}
    """
