from typing import Optional, List

from ..low_level_context import LowLevelContext


class TextContext(LowLevelContext):
    """Text-based context implementation."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["LowLevelContext"],
    ):
        super().__init__(context_id, parent_context)

        self.texts: List[str] = []

    # override
    def destruction(self):
        return super().destruction()

    # override
    def get_this_context_len(self) -> int:
        # This is not useful
        return sum([len(text) for text in self.texts])

    def append_text(self, content: str):
        self.texts.append(content)

    def get_current_context_text(self) -> str:
        if len(self.texts) == 0:
            return ""
        return self.texts[-1]

    def get_whole_context_text(self) -> str:
        parent_text = (
            ""
            if self.parent_context is None
            else self.parent_context.get_whole_context_text()
        )
        return parent_text + "".join(self.texts)

    # Text Context doesn't implement the following methods.

    # override
    def get_last_token_id(self) -> int:
        raise NotImplementedError

    # override
    def push_token_id(self, token_id: int):
        raise NotImplementedError
