# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List


from parrot.constants import LATENCY_ANALYZER_RECENT_N


class LatencyAnalyzer:
    """The analyzer collects the latency per-iter of LLM engines,
    and computes several statistics."""

    def __init__(self):
        self._latency_list: List[float] = []

    def add_latency(self, latency: float):
        """Add a latency to the analyzer."""

        self._latency_list.append(latency)

    def get_average_latency(self) -> float:
        """Get the average latency of the top-n latest latency data."""

        n = LATENCY_ANALYZER_RECENT_N
        actual_n = min(n, len(self._latency_list))

        if actual_n == 0:
            return 0.0

        return sum(self._latency_list[-actual_n:]) / actual_n
