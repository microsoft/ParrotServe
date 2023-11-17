# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="release",
)

qa_func = vm.import_function(
    function_name="qa",
    module_path="app.common",
)


def main():
    print("QA Agent v0.1. Type 'exit' to exit.")

    while True:
        question = input("Your question: ")
        if question == "exit":
            break
        answer = qa_func(question)
        print("Answer: ", answer.get())


vm.run(main)
