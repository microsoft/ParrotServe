from typing import List, Dict
import regex as re
from dataclasses import dataclass

from .placeholder import Placeholder


@dataclass
class FunctionPiece:
    """A piece in the function body."""


@dataclass
class Constant(FunctionPiece):
    """Constant text."""

    text: str


@dataclass
class Prefix(Constant):
    """Prefix in a function."""


@dataclass
class Variable:
    name: str
    is_output: bool


@dataclass
class VariableLoc(FunctionPiece):
    """An input/output location in the function."""

    var: Variable


def parse_func_body(
    body_str: str, args_map: Dict[str, Variable]
) -> List[FunctionPiece]:
    PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]+}}"
    pattern = re.compile(PLACEHOLDER_REGEX)
    iterator = pattern.finditer(body_str)
    last_pos = 0
    ret: List[FunctionPiece] = []

    for match in iterator:
        # Constant
        chunk = body_str[last_pos : match.start()]
        if chunk != "":
            if last_pos == 0:
                ret.append(Prefix(chunk))
            else:
                ret.append(Constant(chunk))

        var_name = body_str[match.start() + 2 : match.end() - 2]
        assert var_name in args_map, f"Parse failed: {var_name} is not defined."
        var = args_map[var_name]
        assert not (
            var.is_output and isinstance(ret[-1], VariableLoc)
        ), "Output loc can't be adjacent to another loc."
        ret.append(VariableLoc(var))

        last_pos = match.end()

    if last_pos < len(body_str):
        ret.append(Constant(body_str[last_pos:]))

    return ret


class ParrotFunction:
    """Parrot function is a simplified abstraction of the "general" semantic function,
    which is used as examples when we play in the Parrot project.

    An example:
        ```
        Tell me a joke about {{topic}}. The joke must contains the
        following keywords: {{keyword}}. The following is the joke: {{joke}}.
        And giving a short explanation to show that why it is funny. The following is the
        explanation for the joke above: {{explanation}}.
        ```
    """

    def __init__(self, name: str, func_body_str: str, func_args: list):
        """For semantic function, function body is just a prompt template.
        After parsed, it turns to be a list of function pieces.
        """

        self.name = name
        self.args: List[Variable] = [
            Variable(name=arg[0], is_output=arg[1]) for arg in func_args
        ]
        self.args_map = dict([(var.name, var) for var in self.args])
        self.body: List[FunctionPiece] = parse_func_body(func_body_str, self.args_map)

    def __call__(self, *args: List[Placeholder], **kwargs: Dict[str, Placeholder]):
        """Calling a parrot function will not execute it immediately.
        Instead, this will submit the call to the executor."""
        pass
