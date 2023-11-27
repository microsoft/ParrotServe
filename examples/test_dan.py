# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="release",
)


dan_request = vm.import_function("dan", "codelib.app.chat")


def main():
    question = "What is the date today?"

    for _ in range(10):
        ack, answer = dan_request(question=question)
        ack_str = ack.get()
        if "DAN: I am waiting for a question" in ack_str:
            print("Verify sucess! ACK=", ack_str)
            print("The answer: ", answer.get())
            continue
        else:
            print("Wrong ACK: ", ack_str)
            print("The answer: ", answer.get())
            continue


vm.run(main)
