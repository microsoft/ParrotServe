# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This module contains functions in chatting senario.

import parrot as P


### Vicuna Chat Functions Start


@P.function(remove_pure_fill=False)
def vicuna_chat_start():
    """A chat between a curious user and an artificial intelligence assistant.
    The assistant gives helpful, detailed, and polite answers to the user's questions.
    """


@P.function()
def vicuna_chat_per_round(
    human_input: P.Input,
    ai_output: P.Output(temperature=0.5, max_gen_length=50),
):
    """
     USER: {{human_input}}
    ASSISTANT: {{ai_output}}
    """


### Vicuna Chat Functions End
