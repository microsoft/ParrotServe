# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, List

from parrot.program.function import SemanticCall
from parrot.utils import get_logger


logger = get_logger("DAG")


class DAGEdge:
    """We use DAGEdge to represent an "Edge" in the data-dependency graph (the Node is the SVPlaceholder).

    An Edge corresponds to a part CallBody, i.e. List[SemanticVariables].
    The part always looks like:

        | Constants | Input Variables | Output Variable |

    A thread may have multiple edges, depending on how many output variables it has.
    """

    def __init__(self, call: Optional[SemanticCall] = None):
        self.call = call

        # A -> B: means A's output is B's input.

        # A "from node" is a "Input Variable" in the edge.
        self.from_nodes: List["SVPlaceholder"] = []
        # And a "to node" is a "Output Variable" in the edge.
        self.to_nodes: List["SVPlaceholder"] = []

    def __repr__(self) -> str:
        if self.call is None:
            return f"DAGEdge()"

        body_list = []
        for idx, edge in self.call.edges_map.items():
            if self == edge:
                body_list.append(str(self.call.func.body[idx]))
        return f'DAGEdge({", ".join(body_list)})'

    def link_with_from_node(self, from_node: "SVPlaceholder"):
        """Link this edge with an from-node."""

        self.from_nodes.append(from_node)
        from_node.out_edges.append(self)

        logger.debug(
            f"Edge {self.call.func.name} links with from-node Placeholder {from_node}"
        )

    def link_with_to_node(self, to_node: "SVPlaceholder"):
        """Link this edge with an out-node."""
        self.to_nodes.append(to_node)
        to_node.in_edges.append(self)

        logger.debug(
            f"Edge {self.call.func.name} links with to-node Placeholder {to_node}"
        )
