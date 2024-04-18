# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Set

from parrot.exceptions import parrot_assert

from .graph import CompletionChain, CompChainGroup
from .nodes import PlaceholderGen
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
    chain: CompletionChain, criteria: PerformanceCriteria, depth: int
) -> None:
    if chain.is_activated:
        return

    # Propagate the performance criteria.
    next_criteria = _back_propagate_criteria(criteria)
    # Grouping chains.
    chain_group = CompChainGroup()

    for node in chain.iter_fill():
        if node.sv.has_producer:
            producer: PlaceholderGen = node.sv.get_producer()
            next_chain: CompletionChain = producer.comp_chain
            next_chain.chain_groups.append(chain_group)
            chain_group.chains.add(next_chain)
            _traverse(next_chain, next_criteria, depth + 1)

    # Lastly, activate the chain.
    chain.activate(criteria, depth)


def activate_completion_chain(
    chain: CompletionChain, criteria: PerformanceCriteria
) -> None:
    """Activates the CompletionChain and propagates the performance deduction result.

    Args:
        chain: The CompletionChain to be activated.
        criteria: The PerformanceCriteria to be assigned to the chain.
    """

    parrot_assert(not chain.is_activated, "Chain is already activated.")

    _traverse(chain=chain, criteria=criteria, depth=0)
