# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Optional

from parrot.exceptions import parrot_assert

from graph.completion_chain import CompletionChain

from .chunked_request import (
    ChunkedRequest,
    RequestBodyChunk,
    TextChunk,
    RequestPlaceholder,
)


class GlobalPrefixMatcher:
    """Global prefix cache will match the first constant text in the request to the prefix.

    If there is a common prefix part between them, we will add the count of the common part.
    If the count reaches a certain threshold, we will consider it as a GlobalPrefix.
    """

    _START_LEN = 40
    _GP_THRESHOLD = 3

    def __init__(self):
        # NOTE(chaofan): This table is two-level.
        # The first level is the first _START_LEN characters of the prefix, to speed up the lookup.
        # The second level is a list of prefix strings.
        # TODO(chaofan): Advanced evict policy.

        self._prefix_counter: Dict[str, Dict[str, int]] = {}

    def add_prefix(self, prefix: str) -> None:
        """Add a prefix to the global prefix cache."""

        # Too short
        if len(prefix) <= self._START_LEN:
            return

        lookup = prefix[: self._START_LEN]
        if lookup not in self._prefix_counter:
            self._prefix_counter[lookup] = {}

        prefixes = list(self._prefix_counter[lookup].keys())

        for k in prefixes:
            # Matched
            if k[self._START_LEN] == prefix[self._START_LEN]:
                # Reduce to common prefix
                i = self._START_LEN
                while i < len(k) and i < len(prefix) and k[i] == prefix[i]:
                    i += 1

                new_k = k[:i]

                if i == len(k):
                    # Common prefix is the same
                    self._prefix_counter[lookup][k] += 1
                else:
                    # Common prefix changes
                    self._prefix_counter[lookup][new_k] = (
                        self._prefix_counter[lookup][k] + 1
                    )
                    self._prefix_counter[lookup].pop(k)
                return

        # Add to table
        self._prefix_counter[lookup][prefix] = 1

    def query_prefix(self, prefix: str) -> int:
        """Query whether the prefix is a global prefix.

        Returns:
            -1 if the prefix is not a global prefix.
            Otherwise, returns the position of the last matched character.
        """

        if len(prefix) <= self._START_LEN:
            return -1

        lookup = prefix[: self._START_LEN]

        if lookup not in self._prefix_counter:
            return -1

        for k, v in self._prefix_counter[lookup].items():
            if v > self._GP_THRESHOLD and prefix.startswith(k):
                return len(k)

        return -1
