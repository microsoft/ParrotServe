# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import re
from typing import List, Dict, Set, Optional

from parrot.exceptions import parrot_assert, ParrotOSUserError

from ..sv.chunked_request import (
    TextChunk,
    PlaceholderNameChunk,
    RequestPlaceholder,
    RequestMetadata,
    ChunkedRequest,
)
from .graph import (
    BaseNode,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    ComputeGraph,
    CompletionChain,
)


class RequestChain:
    """RequestChain is a middle representation of the parsed request, in the form of a chain in
    the graph. It consists a list of Nodes (which is directly compatible in ComputeGraph).

    It's converted from ChunkedRequest (see sv/chunked_request.py).

    It can be inserted into a graph directly.
    """

    def __init__(self) -> None:
        self.nodes: List[BaseNode] = []
        self.chains: List[CompletionChain] = []

        # Only valid after inserted into a graph.
        self.graph: Optional[ComputeGraph] = None
        self._placeholder_mapping: List[Dict] = []

    def __repr__(self) -> str:
        return f"nodes: {self.nodes}, " f"chains: {self.chains}"

    def pretty_print(self) -> str:
        """Pretty print it using Graph's pretty print APIs."""

        ret = "Nodes: \n"
        for node in self.nodes:
            ret += node.pretty_print()

        ret += "Metadata: \n" + str(self.chains[0].request_metadata) + "\n"

        return ret

    @classmethod
    def from_chunked_request(cls, chunked_request: ChunkedRequest) -> "RequestChain":
        """Convert a ChunkedRequest into a RequestChain."""

        request_chain = cls()
        completion_chain_buffer: List[BaseNode] = []

        prev_node: Optional[BaseNode] = None
        for chunk in chunked_request.body:
            is_gen: bool = False

            if isinstance(chunk, TextChunk):
                node = ConstantFill(constant_text=chunk.text)
            elif isinstance(chunk, PlaceholderNameChunk):
                placeholder = chunked_request.placeholders_map[chunk.name]
                if placeholder.is_output:
                    node = PlaceholderGen(placeholder=placeholder)
                    is_gen = True
                else:
                    node = PlaceholderFill(placeholder=placeholder)
            else:
                raise ParrotOSUserError(ValueError("Unknown chunk type."))

            request_chain.nodes.append(node)
            completion_chain_buffer.append(node)

            # Link edge type A with previous node.
            if prev_node is not None:
                prev_node.edge_a_next_node = node
                node.edge_a_prev_node = prev_node
            prev_node = node

            # If current node is Gen, create a new CompletionChain.
            if is_gen:
                completion_chain = CompletionChain.from_nodes(
                    nodes=completion_chain_buffer,
                    request_metadata=chunked_request.metadata,
                )
                request_chain.chains.append(completion_chain)
                completion_chain_buffer = []

        return request_chain

    @property
    def inserted(self) -> bool:
        return self.graph is not None

    def get_placeholder_mapping(self) -> List[Dict]:
        """Get the placeholder mapping after inserted into a graph."""

        parrot_assert(
            self.inserted,
            "Get placeholder mapping failed: RequestChain has not been inserted into a graph.",
        )
        return self._placeholder_mapping

    def insert_to_graph(self, graph: ComputeGraph) -> None:
        """Insert the parsed RequestChain into a specific graph.

        After inserted, placeholder mapping can be fetched from this object. Placeholder mapping
        records the mapping between placeholders and the corresponding semantic variables.
        """

        parrot_assert(
            not self.inserted,
            "Insert failed: RequestChain has been inserted into a graph.",
        )

        for node in self.nodes:
            graph.insert_node(node)

            parrot_assert(node.sv is not None, "Insert failed: SV is not created.")
            if node.has_placeholder():
                placeholder: RequestPlaceholder = node.placeholder

                self._placeholder_mapping.append(
                    {
                        "placeholder_name": placeholder.name,
                        "is_output": placeholder.is_output,
                        "var_name": node.sv_name,
                        "var_id": node.sv_id,
                    }
                )

        self.graph = graph
