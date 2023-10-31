# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# The Vicuna chat template is from:
#   https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

# FIXME(chaofan): Vicuna-13b-v1.3 has strange behavior (Why speaking Chinese?)
# 2023.9.26: Fixed
# 2023.10.23: TODO: Support stateful call in V2
# 2023.10.31: Implemented.

import parrot as P


vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="release",
)


@P.function(remove_pure_fill=False)
def chat_start():
    """A chat between a curious user and an artificial intelligence assistant.
    The assistant gives helpful, detailed, and polite answers to the user's questions.
    """


@P.function()
def chat_per_round(
    human_input: P.Input,
    ai_output: P.Output(temperature=0.5, max_gen_length=50),
):
    """
     USER: {{human_input}}
    ASSISTANT: {{ai_output}}
    """


async def main():
    print("---------- Chatbot v0.1 ----------\n")

    print("Initializing...")
    chat_start.invoke_statefully(context_successor=chat_per_round)
    print("Initialized.")
    print("Hello, How can I assist you today? (Type 'exit' to exit.)")

    while True:
        human_input = input("[HUMAN]: ")
        if human_input == "exit":
            break

        ai_output = chat_per_round.invoke_statefully(
            context_successor=chat_per_round,
            human_input=human_input,
        )
        print(f"[AI]: {ai_output.get()}")

    print("Bye.")


vm.run(main())
