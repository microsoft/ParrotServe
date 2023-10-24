from typing import Optional

from ..low_level_context import LowLevelContext


class TextContext(LowLevelContext):
    """Text-based context implementation."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["LowLevelContext"],
    ):
        super().__init__(context_id, parent_context)

        self._text = ""

    # override
    def get_this_context_len(self) -> int:
        return len(self._text)

    def fill(self, content: str):
        self._text += content

    def get_context_text(self) -> str:
        return self._text

    # Text Context doesn't implement the following methods.

    # override
    def get_last_token_id(self) -> int:
        raise NotImplementedError

    # override
    def push_token_id(self, token_id: int):
        raise NotImplementedError
