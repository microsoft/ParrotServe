# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Set, List, Optional

from parrot.exceptions import parrot_assert

from .graph import CompletionChain, CompChainGroup
from .node import PlaceholderGen, NativeFuncNode, SVProducer
from .semantic_variable import SemanticVariable
from .perf_criteria import PerformanceCriteria


"""Traverses the graph backward, activates related CompletionChains and propagates important 
information.

It provides a primitive function in Parrot system: Performance deduction.

The algorithm works as follows:
1. For a given CompletionChain, it activates the chain and assigns the PerformanceCriteria to it.
2. Then it traverses backward to its predecessors, activates them and propagates the performance
    deduction result recursively.
3. Then algorithm ends when it reaches the end of the graph or an activated node.
"""


def _back_propagate_criteria(criteria: PerformanceCriteria) -> PerformanceCriteria:
    if criteria == PerformanceCriteria.LATENCY:
        return PerformanceCriteria.LATENCY
    elif criteria == PerformanceCriteria.THROUGHPUT:
        return PerformanceCriteria.THROUGHPUT
    else:
        raise NotImplementedError(f"PerformanceCriteria {criteria} is not supported.")


def _traverse(
    var: SemanticVariable,
    criteria: PerformanceCriteria,
    chain_group: Optional[CompChainGroup] = None,
) -> int:  # Return the depth of the variable.
    # Propagate the performance criteria.
    next_criteria = _back_propagate_criteria(criteria)

    # Current depth
    depth = 0

    # Producer of current variable.
    producer = var.get_producer()
    if producer is None:
        return 0

    if isinstance(producer, PlaceholderGen):
        chain: CompletionChain = producer.comp_chain

        if var.is_activated:
            return var.depth

        # Add the chain to chain group.
        if chain_group is not None:
            chain_group.chains.add(chain)
            chain.chain_groups.append(chain_group)

        # Group for the chains in the previous layer.
        prev_chain_group = CompChainGroup()

        for node in chain.iter_fill():
            prev_depth = _traverse(node.sv, next_criteria, prev_chain_group)
            depth = max(depth, prev_depth + 1)

        if chain.first_node.has_edge_a_prev_node:
            prev_gen = chain.first_node.get_edge_a_prev_node()
            parrot_assert(prev_gen.is_gen, "The previous node is not a Gen node.")
            prev_depth = _traverse(prev_gen.sv, next_criteria, prev_chain_group)
            depth = max(depth, prev_depth + 1)
    else:
        parrot_assert(
            isinstance(producer, NativeFuncNode), "Producer is not a NativeFuncNode."
        )

        for in_var in producer.input_vars.values():
            prev_depth = _traverse(in_var, next_criteria, chain_group)
            depth = max(depth, prev_depth)

    # Lastly, activate the variable
    var.activate(criteria, depth)

    return depth


def activate_sv(var: SemanticVariable, criteria: PerformanceCriteria) -> None:
    """Activates a SV and propagates the performance deduction result.

    Args:
        var: The SV to be activated.
        criteria: The PerformanceCriteria to be assigned to the chain.
    """

    parrot_assert(not var.is_activated, "Variable is already activated.")
    _traverse(var, criteria=criteria)


_traverse
