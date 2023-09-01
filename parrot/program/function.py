from typing import List

from .placeholder import Placeholder


class ParrotFunction:
    """Parrot function is a simple version of semantic function, which is
    used as examples when we playing in the Parrot project.

    An example:

        ```
        Tell me a joke about {{input:: topic}}. The joke must contains the
        following keywords: {{input:: keyword}}. The following is the joke: {{output:: joke}}.
        And giving a short explanation to show that why it is funny. The following is the
        explanation for the joke above: {{output:: explanation}}.
        ```
    """

    def __init__(self, func_body: str):
        """For semantic function, function body is just a prompt template."""

    def __call__(
        self, outputs: List[Placeholder], inputs: List[Placeholder], blocking=True
    ):
        """To call a parrot function, the inputs and outputs should all be
        placeholders, as it is needed in variable-level asynchronization."""
        pass


def parrot_function():
    """A decorator for users to define parrot functions."""
    pass
