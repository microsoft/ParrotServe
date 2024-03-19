# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Optional

from parrot.exceptions import parrot_assert

from graph.completion_chain import CompletionChain


class PrefixCache:
    """Local prefix cache will match the prefix according to prefix hash.

    A prefix hash consists of constant part and variable part. The constant part is directly
    hashed according to text content, and the variable part is hashed according to the semantic
    variable id.
    """

    _VARIABLE_SPECIAL_CHAR = "$"

    def __init__(self):
        # Prefix hash -> context id
        self._prefix_map: Dict[str, int] = {}

    def cache_prefix(self, completion_chain: CompletionChain) -> None:
        """NOTE(chaofan): Only the first completion chain is worth caching."""

    def remove_sv(self, sv_id: str) -> None:
        """Remove all prefixes that contain the semantic variable."""

    def prefix_match(
        self,
    ) -> Optional[str]:
        """Match request to the prefix cache."""

        prefix_str = ""

        # last_matched: Optional[str] = None

        # for _, chunk in enumerate(request.body):
        #     if isinstance(chunk, TextChunk):
        #         prefix_str += chunk.text
        #     elif isinstance(chunk, RequestPlaceholder) and chunk.has_var:
        #         prefix_str += (
        #             self._VARIABLE_SPECIAL_CHAR
        #             + chunk.var_id
        #             + self._VARIABLE_SPECIAL_CHAR
        #         )
        #     else:
        #         break

        #     if prefix_str in self._prefix_sv_map:
        #         last_matched = self._prefix_sv_map[prefix_str]

        # if last_matched:
        #     return last_matched
