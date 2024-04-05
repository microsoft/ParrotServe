# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import re
from typing import List, Dict, Set, Optional, Union

from parrot.exceptions import parrot_assert, ParrotCoreUserError
from parrot.utils import RecyclePool

from .chunked_request import (
    TextChunk,
    PlaceholderNameChunk,
    RequestPlaceholder,
    RequestMetadata,
    ChunkedRequest,
)
from .nodes import BaseNode, ConstantFill, PlaceholderFill, PlaceholderGen


"""Data structures for a set of nodes in Graph."""


class _CompletionChainIterator:

    def __init__(self, first_node: BaseNode) -> None:
        self.cur_node = first_node

    def __iter__(self) -> "_CompletionChainIterator":
        return self

    def __next__(self) -> BaseNode:
        if self.cur_node is None or self.cur_node.edge_a_prev_node.is_gen:
            raise StopIteration
        else:
            ret = self.cur_node
            self.cur_node = self.cur_node.edge_a_next_node
            return ret


class _CompletionChainFillIterator:

    def __init__(self, first_node: BaseNode) -> None:
        self.cur_node = first_node

    def __iter__(self) -> "_CompletionChainFillIterator":
        return self

    def __next__(self) -> Union[ConstantFill, PlaceholderFill]:
        if self.cur_node.is_gen:
            raise StopIteration
        else:
            ret = self.cur_node
            self.cur_node = self.cur_node.edge_a_next_node
            return ret


class CompletionChain:
    """A CompletionChain is the basic unit of scheduling (a.k.a Task).

    It contains several Fill primitives and one Gen primitive.

    Fill -> Fill -> Fill -> Gen.
    """

    def __init__(
        self,
        request_chain: "RequestChain",
        first_node: BaseNode,
        gen_node: Optional[PlaceholderGen],
    ) -> None:
        self.request_chain = request_chain
        self.first_node = first_node
        self.gen_node = gen_node

    def iter(self) -> _CompletionChainIterator:
        return _CompletionChainIterator(self.first_node)

    def iter_fill(self) -> _CompletionChainFillIterator:
        return _CompletionChainFillIterator(self.first_node)

    def get_producers(self) -> List[BaseNode]:
        """Get the list of Fill nodes in the chain."""

        return [
            node.edge_b_prev_node
            for node in self.iter()
            if node.edge_b_prev_node is not None
        ]


class _RequestChainIterator:

    def __init__(self, first_node: BaseNode) -> None:
        self.cur_node = first_node

    def __iter__(self) -> "_RequestChainIterator":
        return self

    def __next__(self) -> BaseNode:
        if self.cur_node is None:
            raise StopIteration
        else:
            ret = self.cur_node
            self.cur_node = self.cur_node.edge_a_next_node
            return ret


class RequestChain:
    """RequestChain is a middle representation of the parsed request, in the form of a chain in
    the graph. It consists a list of Nodes (which is directly compatible in ComputeGraph).

    It's converted from ChunkedRequest (see sv/chunked_request.py).

    It can be inserted into a graph directly.
    """

    def __init__(self, metadta: RequestMetadata) -> None:
        self.first_node: Optional[BaseNode] = None
        self.completion_chains: List[CompletionChain] = []
        self.metadata = metadta

        # Flags
        self.sv_created = False
        self.inserted = False

        # Only valid after inserted into a graph.
        self.placeholder_mapping: List[Dict] = []

    def iter(self) -> _RequestChainIterator:
        return _RequestChainIterator(self.first_node)

    def __repr__(self) -> str:
        return f"RequestChain(first_node={self.first_node})"

    def pretty_print(self) -> str:
        """Pretty print it using Graph's pretty print APIs."""

        ret = "Nodes: \n"
        for node in self.iter():
            ret += node.pretty_print()

        ret += "Metadata: \n" + str(self.metadata) + "\n"

        return ret

    @classmethod
    def from_chunked_request(cls, chunked_request: ChunkedRequest) -> "RequestChain":
        """Convert a ChunkedRequest into a RequestChain."""

        request_chain = cls(chunked_request.metadata)
        prev_node: Optional[BaseNode] = None
        completion_chain_first_node: Optional[BaseNode] = None

        for i, chunk in enumerate(chunked_request.body):
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
                raise ParrotCoreUserError(ValueError("Unknown chunk type."))

            # Record first node
            if i == 0:
                request_chain.first_node = node
                completion_chain_first_node = node

            # Link edge type A with previous node.
            if prev_node is not None:
                prev_node.edge_a_next_node = node
                node.edge_a_prev_node = prev_node
            prev_node = node

            # If current node is Gen, create a new CompletionChain.
            if is_gen:
                completion_chain = CompletionChain(
                    request_chain=request_chain,
                    first_node=completion_chain_first_node,
                    gen_node=node,
                )
                request_chain.completion_chains.append(completion_chain)
                completion_chain_first_node = node.edge_a_next_node

        return request_chain

    def get_placeholder_mapping(self) -> List[Dict]:
        """Get the placeholder mapping after inserted into a graph."""

        parrot_assert(
            self.inserted,
            "Get placeholder mapping failed: RequestChain has not been inserted into a graph.",
        )
        return self.placeholder_mapping


class ComputeGraph:
    """Computational graph of LLM requests linked by Semantic Variables.

    It's made up of a list of nodes (And edges are maintained by nodes and SVs).

    It has several properties:
    1. It's a DAG (Directed Acyclic Graph) i.e. topologically sorted (if all requests are created valid).
       Thus, we can schedule it in a topological order.
    2. When scheduling, only chains are enterring and leaving the graph.
    3. Every node's in-degree is at most 2 (1 type A edge + 1 type B edge). Out-degree is not limited.
    """

    def __init__(self) -> None:
        self.nodes: Set[BaseNode] = set()
        self.chains: List[CompletionChain] = []

        self.node_id_pool = RecyclePool("Node Pool")

    def _insert_node(self, node: BaseNode) -> None:
        self.nodes.add(node)
        node.id_in_graph = self.node_id_pool.allocate()

        # Link edge type B
        if node.is_gen:
            node.sv.assign_producer(node)
        else:
            node.sv.add_consumer(node)

    def insert_and_update_request_chain(self, request_chain: RequestChain) -> None:
        """Insert a RequestChain into the graph, and update its info.

        After inserted, placeholder mapping can be fetched from this object. Placeholder mapping
        records the mapping between placeholders and the corresponding semantic variables.
        """

        parrot_assert(
            request_chain.sv_created,
            "Insert failed: SV should be created before inserting into a graph.",
        )

        parrot_assert(
            not request_chain.inserted,
            "Insert failed: RequestChain has been inserted into a graph.",
        )

        for node in request_chain.iter():
            self._insert_node(node)

            parrot_assert(node.sv is not None, "Insert failed: SV is not created.")
            if node.has_placeholder():
                placeholder: RequestPlaceholder = node.placeholder

                # Maintain the placeholder mapping
                request_chain.placeholder_mapping.append(
                    {
                        "placeholder_name": placeholder.name,
                        "is_output": placeholder.is_output,
                        "var_name": node.sv_name,
                        "var_id": node.sv_id,
                    }
                )
        self.chains.extend(request_chain.completion_chains)

        request_chain.inserted = True

    def remove_completion_chain(self, completion_chain: CompletionChain) -> None:
        """Remove a CompletionChain from the graph. This is called when the task is finished."""

        self.chains.remove(completion_chain)
        for node in completion_chain.iter():
            self.nodes.remove(node)
            self.node_id_pool.free(node.id_in_graph)

            # Remove edge type B
            if node.is_gen:
                node.sv.remove_producer()
            else:
                node.sv.remove_consumer(node)
