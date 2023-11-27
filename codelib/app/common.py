# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# This module contains functions that used in the common daily life and workflow.

import parrot as P


@P.function(conversation_template=P.vicuna_template)
def tell_me_a_joke(
    topic: P.Input,
    topic2: P.Input,
    joke: P.Output,
    explanation: P.Output(P.SamplingConfig(temperature=0.5)),
):
    """Tell the me a joke about {{topic}} and {{topic2}}. {{joke}}.
    Good, then giving a short explanation to show that why it is funny.
    The explanation should be short, concise and clear. {{explanation}}.
    """


@P.function(formatter=P.allowing_newline, cache_prefix=False)
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


@P.function(formatter=P.allowing_newline)
def qa(
    question: P.Input,
    answer: P.Output,
):
    """You are a helpful assistant who can answer questions. For each question, you
    should answer it correctly and concisely. And try to make the answer as short as possible (Ideally,
    just one or two words).

    The question is: {{question}}.

    The answer is: {{answer}}.
    """
