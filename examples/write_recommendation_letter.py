# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This application is a generator of recommendation letters.
# Given the basic information of a student, it can generate a recommendation letter for him/her.
# But we don't recommend you to use it in real life, if you are really a professor !!!

import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="debug",
)

letter_generator = vm.import_function("write_recommendation_letter", "codelib.app.common")


def main():
    letter = letter_generator(
        stu_name="John",
        prof_name="Prof. Smith",
        major="Computer Science",
        grades="3.8",
        specialty="Basketball. Good at playing basketball. Used to be team leader of the school basketball team.",
    )

    letter_str = letter.get()
    print("\n\n ---------- RECOMMEND LETTER ---------- ")
    print(letter_str)


vm.run(main)
