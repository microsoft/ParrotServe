# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)
# The Vicuna chat template is from:
#   https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

import parrot as P
import aioconsole  # We use aioconsole to read input asynchronously


vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.init()


@P.function(caching_prefix=False)
def chat_per_round(
    human_input: P.Input,
    ai_output: P.Output,
):
    # fmt: off
    """ USER: {{human_input}} ASSISTANT: {{ai_output}}"""
    # fmt: on


chat_ctx = P.shared_context("vicuna_7b_v1.3_local")


def chatbot_init():
    chat_ctx.fill(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )


async def main():
    chatbot_init()
    print("---------- Chatbot v0.1 ----------\n")

    print(chat_per_round.body)

    while True:
        human_input = P.placeholder()
        ai_output = P.placeholder()

        with chat_ctx.open("w") as handler:
            handler.call(chat_per_round, human_input, ai_output)

        human_input_str = await aioconsole.ainput("[HUMAN]: ")
        if human_input_str == "exit":
            break

        human_input.assign(human_input_str)

        ai_output_str = await ai_output.get()
        print(f"[AI]: {ai_output_str}")

    print("Bye.")
    chat_ctx.free()


vm.run(main())
