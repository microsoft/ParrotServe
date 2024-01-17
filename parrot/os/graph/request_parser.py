# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import re
from typing import List, Dict, Set

from parrot.exceptions import parrot_assert

from .request import RequestPlaceholder
from .gen_task import GenTask, TaskMetadata
from .graph import (
    BaseNode,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
)


class ParsedRequest:
    """A parsed request."""

    def __init__(self):
        self.nodes: List[BaseNode] = []
        self.gen_tasks: List[GenTask] = []
        self.node_idx_to_placeholders: Dict[int, RequestPlaceholder] = {}


class RequestParser:
    """Request Parser: parse a request into GenTasks and BaseNodes."""

    def parse(self, payload: Dict) -> ParsedRequest:
        """Parse the request into GenTasks and BaseNodes."""

        # Pre-defined regex of placeholder name.
        PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]*}}"

        # Get arguments from payload packet.
        template: str = payload["template"]
        remove_pure_fill: bool = payload["remove_pure_fill"]
        placeholders: Dict = payload["placeholders"]

        # Outputs.
        nodes: List[BaseNode] = []
        node_idx_to_placeholders: Dict[int, RequestPlaceholder] = {}
        result = ParsedRequest()

        # Step 1. Parse placeholders.
        placeholders_map: Dict[str, RequestPlaceholder] = {}
        for placeholder_name, placeholder in placeholders.items():
            parrot_assert(
                placeholder_name not in placeholders_map, "Duplicate placeholder name."
            )
            placeholders_map[placeholder_name] = RequestPlaceholder(**placeholder)

        # Step 2. Parse prompt body.
        pattern = re.compile(PLACEHOLDER_REGEX)
        iterator = pattern.finditer(template)
        outputs: Set[str] = set()
        last_pos = 0

        for matched in iterator:
            # Constant
            chunk = template[last_pos : matched.start()]
            if chunk != "":
                nodes.append(ConstantFill(constant_text=chunk))

            # Placeholder
            placeholder_name = template[matched.start() + 2 : matched.end() - 2]
            parrot_assert(
                placeholder_name in placeholders_map
            ), f'Parse failed: Placeholder name "{placeholder_name}" is not defined.'

            placeholder = placeholders_map[placeholder_name]
            node_idx_to_placeholders[len(nodes)] = placeholder
            if placeholder.is_output:
                assert (
                    not placeholder.name in outputs
                ), "Output param can't be used twice."
                outputs.add(placeholder.name)
                nodes.append(
                    PlaceholderGen(sampling_config=placeholder.sampling_config)
                )
            else:
                nodes.append(PlaceholderFill())

            last_pos = matched.end()

        if remove_pure_fill:
            # NOTE(chaofan): we prune all pieces after the last output.
            # The following code is also correct for no output case.

            while not isinstance(nodes[-1], PlaceholderGen):
                nodes.pop()

        # Step 3. Create GenTasks.
        metadata = TaskMetadata(**payload.fromkeys(TaskMetadata.TASK_METADATA_KEYS))

        result.nodes.extend(nodes)
        result.node_idx_to_placeholders.update(node_idx_to_placeholders)

        for i in range(len(nodes) - 1):
            result.nodes[i].edge_a_next_node = result.nodes[i + 1]
            result.nodes[i + 1].edge_a_prev_node = result.nodes[i]

        prev_fill_nodes: List[BaseNode] = []
        for node in nodes:
            if isinstance(node, PlaceholderFill):
                prev_fill_nodes.append(node)
            elif isinstance(node, PlaceholderGen):
                result.gen_tasks.append(
                    GenTask(
                        task_metadata=metadata,
                        fill_nodes=prev_fill_nodes,
                        gen_node=node,
                    )
                )
                prev_fill_nodes = []
