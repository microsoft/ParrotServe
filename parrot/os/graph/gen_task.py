# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
from dataclasses import dataclass

from .graph import BaseNode
from ..resource.model_type import get_model_type


class TaskMetadata:
    TASK_METADATA_KEYS = ["model_type", "models"]

    def __init__(self, model_type: str, models: List[str]):
        self.model_type = get_model_type(model_type)
        self.models = models


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


class TaskGroup:
    pass
