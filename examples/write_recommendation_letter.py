# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import parrot as P

# We need to start a definition scope before defining any functions.
vm = P.VirtualMachine("configs/vm/single_vicuna_13b_v1.3.json")
vm.init()


# Now we can start to define a "Parrot function".
# The magical thing is that, the function is "defined" by the
# docstring! (in a natural language way)
# The function will be automatically be registered to the environment


@P.function(formatter=P.AllowingNewlineFormatter)
def write_recommendation_letter(
    stu_name: P.Input,
    prof_name: P.Input,
    major: P.Input,
    grades: P.Input,
    specialty: P.Input,
    letter: P.Output,
):
    r"""You are a professor in the university. Please write a recommendation for a student's PhD application.

    Note that the letter should not be too long. You can write at most 300 words. The letter should be written in English, end with "Sincerely, Prof. {{prof_name}}".

    Here are some information of the student:
    Name: {{stu_name}}
    Major: {{major}}
    Grades: {{grades}}/4.0
    Specialty: {{specialty}}

    The following is the letter you should write: {{letter}}
    """


# Then we can start to define the main function.
async def main():
    letter = write_recommendation_letter(
        stu_name="John",
        prof_name="Prof. Smith",
        major="Computer Science",
        grades="3.8",
        specialty="Basketball. Good at playing basketball. Used to be team leader of the school basketball team.",
    )

    letter_str = await letter.get()
    print("\n\n ---------- RECOMMEND LETTER ---------- ")
    print(letter_str)


# Just run it.
vm.run(main())
