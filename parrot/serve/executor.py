# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from parrot.serve.graph import BaseNode
from parrot.serve.backend_repr import Context


class ExecuteUnit:
    """ExecuteUnit wraps BaseNode by adding context binding info."""

    def __init__(self, node: BaseNode, context: Context):
        self.node = node
        self.context = context


class Executor:
    """The executor is responsible for executing task."""
