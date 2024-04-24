# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List

from .func_mutator import (
    FuncMutator,
    SemanticFunction,
    Constant,
    Parameter,
)

from .sequential import Sequential


class PromptFormatter(FuncMutator):
    """Prompt formatter is defined as a mutator which only mutates the Constant pieces."""

    def __init__(self, replace_pairs: List[List[str]]):
        self._replace_pairs = replace_pairs

    def _visit_func(self, func: SemanticFunction) -> SemanticFunction:
        return func

    def _visit_constant(self, constant: Constant) -> Constant:
        constant_str = constant.text
        for src, tgt in self._replace_pairs:
            constant_str = constant_str.replace(src, tgt)
        return Constant(constant.idx, constant_str)

    def _visit_parameter(self, param: Parameter) -> Parameter:
        return param


class PyIndentRemover(PromptFormatter):
    """Python's docstring has indents, which disturbs the prompt. This mutator removes the indents."""

    _possible_indents = ["\t", "    "]  # tab or 4 spaces

    def __init__(self):
        super().__init__([(indent, "") for indent in self._possible_indents])


class SquashIntoOneLine(PromptFormatter):
    """Squash a function body into one line."""

    def __init__(self):
        super().__init__([["\n", " "]])


class AlwaysOneSpace(PromptFormatter):
    """Replace all spaces with one space."""

    def __init__(self):
        super().__init__([[" " * i, " "] for i in range(16, 1, -1)])


standard_formatter = Sequential(
    [
        PyIndentRemover(),
        SquashIntoOneLine(),
        AlwaysOneSpace(),
    ]
)

allowing_newline = Sequential(
    [
        PyIndentRemover(),
        AlwaysOneSpace(),
    ]
)
