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

chat_start = vm.import_function("vicuna_chat_start", "app.chat")
chat_per_round = vm.import_function("vicuna_chat_per_round", "app.chat")


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
