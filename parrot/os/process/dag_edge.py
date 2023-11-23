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

        # An in node is a "Input Variable" in the edge.
        self.in_nodes: List["SVPlaceholder"] = []
        # And an out node is a "Output Variable" in the edge.
        self.out_nodes: List["SVPlaceholder"] = []

    def link_with_in_node(self, in_node: "SVPlaceholder"):
        """Link this edge with an in-node."""

        print(in_node)

        self.in_nodes.append(in_node)
        in_node.out_edges.append(self)

        logger.debug(
            f"Edge {self.call.func.name} links with in-node Placeholder {in_node}"
        )

    def link_with_out_node(self, out_node: "SVPlaceholder"):
        """Link this edge with an out-node."""
        self.out_nodes.append(out_node)
        out_node.in_edges.append(self)

        logger.debug(
            f"Edge {self.call.func.name} links with out-node Placeholder {out_node}"
        )
