# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import re
from typing import List, Dict, Set

from parrot.exceptions import parrot_assert, ParrotOSUserError

from .request import RequestPlaceholder
from .gen_task import GenTask, TaskMetadata
from .graph import BaseNode, ConstantFill, PlaceholderFill, PlaceholderGen, StaticGraph


class ParsedRequest:
    """A parsed request."""

    def __init__(self):
        self.nodes: List[BaseNode] = []
        self.gen_tasks: List[GenTask] = []

    def __repr__(self) -> str:
        return f"Nodes: {self.nodes}\nGenTasks: {self.gen_tasks}"

    def pretty_print(self) -> str:
        """Pretty print the parsed request."""

        ret = "Nodes: \n"
        for node in self.nodes:
            ret += node.pretty_print()

        ret += "GenTasks Metadata: \n"
        for task in self.gen_tasks:
            ret += str(task.task_metadata) + "\n"

        return ret

    def insert_to_graph(self, graph: StaticGraph) -> List[Dict]:
        """Insert the parsed request into a graph. Return a List of SVs info of placeholders."""

        ret = []

        for node in self.nodes:
            graph.insert_node(node)

            parrot_assert(node.sv is not None, "Insert failed: SV is not created.")
            if node.has_placeholder():
                placeholder: RequestPlaceholder = node.placeholder

                ret.append(
                    {
                        "placeholder_name": placeholder.name,
                        "is_output": placeholder.is_output,
                        "var_name": node.sv_name,
                        "var_id": node.sv_id,
                    }
                )

        return ret


class RequestParser:
    """Request Parser: parse a request into GenTasks and BaseNodes."""

    def _preprocess(self, payload: Dict) -> Dict:
        """Preprocess the payload packet. This will do the format check and assign default values."""

        # Check format.
        parrot_assert("template" in payload, "Missing field 'template' in request.")
        parrot_assert(
            "placeholders" in payload, "Missing field 'placeholders' in request."
        )

        processed_payload = payload.copy()

        # Assign default values.
        processed_payload.setdefault("models", [])
        processed_payload.setdefault("model_type", "token_id")
        processed_payload.setdefault("remove_pure_fill", True)

        return processed_payload

    def parse(self, payload: Dict) -> ParsedRequest:
        """Parse the request into GenTasks and BaseNodes."""

        payload = self._preprocess(payload)

        # Pre-defined regex of placeholder name.
        PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]*}}"

        # Get arguments from payload packet.
        template: str = payload["template"]
        remove_pure_fill: bool = payload["remove_pure_fill"]
        placeholders: Dict = payload["placeholders"]

        # Outputs.
        nodes: List[BaseNode] = []
        result = ParsedRequest()

        placeholders_map: Dict[str, RequestPlaceholder] = {}
        for placeholder in placeholders:
            try:
                parsed_placeholder = RequestPlaceholder(**placeholder)
            except BaseException as e:
                raise ParrotOSUserError(e)

            placeholder_name = placeholder["name"]
            parrot_assert(
                placeholder_name not in placeholders_map, "Duplicate placeholder name."
            )
            placeholders_map[placeholder_name] = parsed_placeholder

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
                placeholder_name in placeholders_map,
                f'Parse failed: Placeholder name "{placeholder_name}" is not defined.',
            )

            placeholder = placeholders_map[placeholder_name]
            if placeholder.is_output:
                assert (
                    not placeholder.name in outputs
                ), "Output param can't be used twice."
                outputs.add(placeholder.name)
                nodes.append(PlaceholderGen(placeholder=placeholder))
            else:
                nodes.append(PlaceholderFill(placeholder=placeholder))

            last_pos = matched.end()

        if remove_pure_fill:
            # NOTE(chaofan): we prune all pieces after the last output.
            # The following code is also correct for no output case.

            while not isinstance(nodes[-1], PlaceholderGen):
                nodes.pop()

        # Step 3. Create GenTasks.
        payload_metadata = {}
        for key in TaskMetadata.TASK_METADATA_KEYS:
            payload_metadata[key] = payload[key]
        metadata = TaskMetadata(**payload_metadata)

        result.nodes.extend(nodes)

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

        return result
