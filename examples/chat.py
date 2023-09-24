# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)
# The Vicuna chat template is from:
#   https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

from parrot import env, P
import logging

# Disable the logging
logging.disable(logging.DEBUG)
logging.disable(logging.INFO)

env.register_tokenizer("hf-internal-testing/llama-tokenizer")
env.register_engine(
    "vicuna_13b_v1.3_local",
    host="localhost",
    port=8888,
    tokenizer="hf-internal-testing/llama-tokenizer",
)


chat_template = """
USER: {human_input}
ASSISTANT: {ai_output}
"""


@P.function()
def chat(
    chat_history: P.Input,
    human_input: P.Input,
    ai_output: P.Output,
):
    """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    {{chat_history}}
    USER: {{human_input}}
    ASSISTANT: {{ai_output}}
    """


async def main():
    chat_history_str = ""
    print("---------- Chatbot v0.1 ----------\n")

    while True:
        chat_history = P.placeholder()
        human_input = P.placeholder()
        ai_output = P.placeholder()

        chat(chat_history, human_input, ai_output)

        human_input_str = input("[HUMAN]: ")
        if human_input_str == "exit":
            break

        chat_history.assign(chat_history_str)
        human_input.assign(human_input_str)

        ai_output_str = await ai_output.get()
        print(f"[AI]: {ai_output_str}")

        chat_history_str += chat_template.format(
            human_input=human_input_str, ai_output=ai_output_str
        )

        # print("Chat history: ", chat_history_str)


env.parrot_run_aysnc(main())
