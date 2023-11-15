# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from parrot.program.function import SemanticCall


class DAGNode:
    """For a call/native code, it may wait for multiple placeholders.

    We use DAGNode to represent the call/native code, and use the edges to represent
    the placeholders.
    """

    def __init__(self, call: Optional[SemanticCall] = None):
        self.call = call
        self._in_degree = 0

    def add_in_edge(self):
        self._in_degree += 1

    def remove_in_edge(self):
        self._in_degree -= 1

    @property
    def in_degree(self):
        return self._in_degree

    def reset(self):
        self._in_degree
