# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional
from abc import ABC, abstractmethod

from parrot.utils import get_logger


logger = get_logger("LowLevelContext")


class LowLevelContext(ABC):
    """Base class for low-level implementation of Context."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["LowLevelContext"],
    ):
        self.context_id = context_id
        self.sub_context_ids: List[int] = []

        # Link with parent context
        self.parent_context = parent_context
        if self.parent_context is not None:
            parent_context.sub_context_ids.append(self.context_id)

    def destruction(self):
        """Destruct the context. If we call this function, the context obj should not be used
        anymore."""

        if self.parent_context is not None:
            self.parent_context.sub_context_ids.remove(self.context_id)
        assert (
            len(self.sub_context_ids) == 0
        ), f"Sub-contexts {self.sub_context_ids[0]} should be deleted first."

    def get_context_len(self) -> int:
        """Return the length of the context."""

        parent_len = self.parent_context.get_context_len() if self.parent_context else 0
        return parent_len + self.get_this_context_len()

    @abstractmethod
    def get_this_context_len(self) -> int:
        """Return the length of the context, without recursing into parent contexts."""

    # The following methods are used in the token-level context.

    @abstractmethod
    def push_token_id(self, token_id: int):
        """Push a token id to the context."""

    @abstractmethod
    def get_last_token_id(self) -> int:
        """Return the last token id."""
