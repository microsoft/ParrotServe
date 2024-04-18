# Copyright (c) 2023 by Microsoft Corporation.
# Author: Zhenhua Han (zhenhua.han@microsoft.com); Chaofan Lin (v-chaofanlin@microsoft.com)

# This module contains a multi-agents collaboration task: developing a snake game.
# This is done by a role-playing prompt method. The architecture of this workflow is:
#                       Architect
#  Coder (main.py) | Coder (snake.py) | Coder (game.py)
#                       Reviewer
#  Reviser (main.py) | Reviser (snake.py) | Reviser (game.py)


import parrot as P


@P.semantic_function(formatter=P.allowing_newline)
def architect(
    main_api: P.Output,
    game_api: P.Output,
    snake_api: P.Output,
    food_api: P.Output,
):
    """Role: You are a system architect.

    User gives you a task. You design a list of files and design a list of APIs with full function signatures (with functionality as comments) for each file to achieve the task.

    Task: Write a cli snake game in python.

    Response in the format:

    Files:
    main.py
    game.py
    snake.py
    food.py
    ......

    APIs:
    main.py:
    Code:```{{main_api}}```

    game.py:
    Code:```{{game_api}}```

    snake.py:
    Code:```{{snake_api}}```

    food.py:
    Code:```{{food_api}}```
    """


@P.semantic_function(formatter=P.allowing_newline)
def programmer(
    architect_response: P.Input,
    file_name: str,
    other_filename1: str,
    other_filename2: str,
    other_filename3: str,
    code: P.Output,
):
    """Role: You are an expert programmer. You implement the APIs given by the system architect.

    APIs:
    {{architect_response}}

    You only need to implement {{file_name}}. Implement all functions and additional functions you need. DO NOT LET ME TO IMPLEMENT ANYTHING!!!!
    Make sure your response code is runnable.
    Do not response any content in {{other_filename1}}, {{other_filename2}} and {{other_filename3}}. Strictly follow the response format. Do not answer any other content or suggestions.

    Response format:

    ```{{code}}```"""


@P.semantic_function(formatter=P.allowing_newline)
def reviewer(
    main_code: P.Input,
    snake_code: P.Input,
    game_code: P.Input,
    food_code: P.Input,
    review: P.Output,
):
    """Role: You are an expert code reviewer.
    Task:
    You review the code given by the expert programmer and share your comments. Do not write your own code.

    main.py:
    {{main_code}}

    snake.py:
    {{snake_code}}

    game.py:
    {{game_code}}

    food.py:
    {{food_code}}

    Comments:
    {{review}}
    """


@P.semantic_function(formatter=P.allowing_newline)
def reviser(
    main_code: P.Input,
    snake_code: P.Input,
    game_code: P.Input,
    food_code: P.Input,
    file_name: str,
    review: P.Input,
    revised_code: P.Output,
):
    """Codebase:

    main.py:
    {{main_code}}

    snake.py
    {{snake_code}}

    game.py
    {{game_code}}

    food.py
    {{food_code}}

    Review comments:
    {{review}}

    Task: You just implemented ``{{file_name}}`` Given the code and review comments. Revise ``{{file_name}}``. Implement all functions and additional functions you need. DO NOT LET ME TO IMPLEMENT ANYTHING!!!!
    Make sure your response code is runnable.
    Do not response any content in game.py and snake.py. Strictly follow the response format. Do not answer any other content or suggestions.

    Response format:

    ```{{revised_code}}```
    """
