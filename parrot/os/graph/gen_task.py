# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union
from dataclasses import dataclass

from .graph import BaseNode
from ..resource.model_type import get_model_type, ModelType


@dataclass
class TaskMetadata:
    TASK_METADATA_KEYS = ["model_type", "models", "remove_pure_fill"]

    model_type: Union[str, ModelType]
    models: List[str]
    remove_pure_fill: bool

    def __post_init__(self):
        self.model_type = get_model_type(self.model_type)


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
        self.task_metadata = task_metadata
        self.fill_nodes = fill_nodes
        self.gen_node = gen_node

    def is_ready(self) -> bool:
        """Check whether the GenTask is ready to be scheduled. It's ready if all its inputs are ready."""

        if len(self.fill_nodes) == 0:
            return True

        if self.fill_nodes[0].in_degree != 0:
            return False

        for fill_node in self.fill_nodes:
            if fill_node.in_degree == 2:
                return False

        return True


class TaskGroup:
    pass
