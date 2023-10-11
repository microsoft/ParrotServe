# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)
# The Vicuna chat template is from:
#   https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

# FIXME(chaofan): Vicuna-13b-v1.3 has strange behavior (Why speaking Chinese?)
# 2023.9.26: Fixed

import parrot as P


vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.launch()


@P.function(caching_prefix=False)
def chat_per_round(
    human_input: P.Input,
    ai_output: P.Output,
):
    """
    USER: {{human_input}}
    ASSISTANT: {{ai_output}}
    """


chat_ctx = P.shared_context("vicuna_13b_v1.3_local")


def chatbot_init():
    chat_ctx.fill(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )


async def main():
    chatbot_init()
    print("---------- Chatbot v0.1 ----------\n")

    while True:
        human_input_str = input("[HUMAN]: ")
        if human_input_str == "exit":
            break

        with chat_ctx.open("w") as handler:
            ai_output = handler.call(chat_per_round, human_input_str)
        print(f"[AI]: {await ai_output.get()}")

    print("Bye.")
    chat_ctx.free()


vm.run(main())
