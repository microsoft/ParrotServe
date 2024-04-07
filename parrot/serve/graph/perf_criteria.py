# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum


class PerformanceCriteria(Enum):
    """Performance criteria for a SemanticVariable.Get behavior."""

    # Optimize latency
    LATENCY = 0

    # Optimize throughput
    THROUGHPUT = 1

    # Time-to-first-token
    TTFT = 2

    # Time-per-output-token
    TPOT = 3


def get_performance_criteria(criteria: str) -> PerformanceCriteria:
    """Get the performance criteria from a string."""

    if criteria == "latency":
        return PerformanceCriteria.LATENCY
    elif criteria == "throughput":
        return PerformanceCriteria.THROUGHPUT
    elif criteria == "TTFT":
        return PerformanceCriteria.TTFT
    elif criteria == "TPOT":
        return PerformanceCriteria.TPOT
    else:
        raise NotImplementedError(f"PerformanceCriteria {criteria} is not supported.")
