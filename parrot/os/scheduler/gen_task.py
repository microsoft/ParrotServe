# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Optional
from dataclasses import dataclass


from ..graph.graph import BaseNode
from ..context.context import Context



class GenTask:
    """A GenTask is the basic unit of scheduling.

    It contains several Fill primitives and one Gen primitive.
    """

    def __init__(
        self,
        task_metadata: TaskMetadata,
        fill_nodes: List[BaseNode],
        gen_node: BaseNode,
    ):
        # Original data
        self.task_metadata = task_metadata
        self.fill_nodes = fill_nodes
        self.gen_node = gen_node

        # Tokenization
        self.tokenized = False
        self.tokenized_fills: Dict[str, List] = {} # {tokenizer_name: tokenized fills}

        # Context binding
        self.context_bound = False
        self.bound_contexts: List[Context] = []

        # Grouped
        self.group: Optional[TaskGroup] = None

    async def wait_ready(self):
        """Wait the GenTask to be ready. It's ready if all its inputs are ready."""

        if len(self.fill_nodes) == 0:
            return

        for fill_node in self.fill_nodes:
            await fill_node.wait_ready()
    
    @property
    def is_grouped(self):
        return self.group is not None


class TaskGroup:
    pass
