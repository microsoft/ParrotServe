# Copyright (c) 2023 by Microsoft Corporation.

# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from parrot import env, P

# We need to configure the environment before we can use it.
# Here we use the OPT model from facebook as an example.

# First, we need to register the tokenizer we want to use.
env.register_tokenizer("facebook/opt-13b")
# Then, we need to register the engine we want to use.
env.register_engine(
    "opt_13b_local",
    host="localhost",
    port=8888,
    tokenizer="facebook/opt-13b",
)


# Now we can start to define a "Parrot function".
# The magical thing is that, the function is "defined" by the
# docstring! (in a natural language way)
# The function will be automatically be registered to the environment


@P.function()
def write_recommendation_latter(
    name: P.Input,
    major: P.Input,
    grades: P.Input,
    specialty: P.Input,
    letter: P.Output,
):
    """You are a professor in the university. Given the basic information about a student including his name, major, grades and specialty, please write a recommendation for his PhD application.

    Here are the information of the student:
    Name: {{name}}
    Major: {{major}}
    Grades: {{grades}}/4.0
    Specialty: {{specialty}}

    The following is the letter you should write: {{letter}}
    """


# Now we can call the function we just defined.


async def main():
    # First we need some placeholders.
    name = P.placeholder()
    major = P.placeholder()
    grades = P.placeholder()
    specialty = P.placeholder()
    letter = P.placeholder()

    # Then we can call the function.
    write_recommendation_latter(
        name=name,
        major=major,
        grades=grades,
        specialty=specialty,
        letter=letter,
    )

    # Now we can fill in the placeholders.
    name.assign("John")
    major.assign("Computer Science")
    grades.assign("3.8")
    specialty.assign(
        "Basketball. Good at playing basketball. Used to be team leader of the school basketball team."
    )

    letter_str = await letter.get()
    print("\n\n ---------- RECOMMEND LETTER ---------- ")
    print(letter_str)


env.parrot_run_aysnc(main())
