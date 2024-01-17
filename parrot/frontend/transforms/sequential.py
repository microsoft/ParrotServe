# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List

from .func_mutator import FuncMutator, SemanticFunction


class Sequential:
    """Sequential transforms for a program."""

    def __init__(self, transforms: List[FuncMutator]):
        self._transforms = transforms

    def transform(self, func: SemanticFunction) -> SemanticFunction:
        """Transform a function and return a new function."""

        for transform in self._transforms:
            func = transform.transform(func)

        return func
