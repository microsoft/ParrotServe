# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from parrot import env, P
import time

# We need to configure the environment before we can use it.
# Here we use the Vicuna model from LMSYS as an example.

# First, we need to register the tokenizer we want to use.
env.register_tokenizer("hf-internal-testing/llama-tokenizer")
# Then, we need to register the engine we want to use.
env.register_engine(
    "vicuna_13b_v1.3_local",
    host="localhost",
    port=8888,
    tokenizer="hf-internal-testing/llama-tokenizer",
)


# Now we can start to define a "Parrot function".
# The magical thing is that, the function is "defined" by the
# docstring! (in a natural language way)
# The function will be automatically be registered to the environment


@P.function()
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


# Now we can call the function we just defined.


async def main():
    # First we need some placeholders.
    stu_name = P.placeholder()
    prof_name = P.placeholder()
    major = P.placeholder()
    grades = P.placeholder()
    specialty = P.placeholder()
    letter = P.placeholder()

    # Then we can call the function.
    write_recommendation_letter(
        stu_name=stu_name,
        prof_name=prof_name,
        major=major,
        grades=grades,
        specialty=specialty,
        letter=letter,
    )

    # To monitor the caching tokens statistics
    time.sleep(20)

    # Now we can fill in the placeholders.
    stu_name.assign("John")
    prof_name.assign("Prof. Smith")
    major.assign("Computer Science")
    grades.assign("3.8")
    specialty.assign(
        "Basketball. Good at playing basketball. Used to be team leader of the school basketball team."
    )

    letter_str = await letter.get()
    print("\n\n ---------- RECOMMEND LETTER ---------- ")
    print(letter_str)


env.parrot_run_aysnc(main())
