# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional, List, Literal
from dataclasses import dataclass

from .low_level_context import LowLevelContext


@dataclass
class TextChunk:
    text: str
    role: Literal["user", "assistant"]


class TextContext(LowLevelContext):
    """Text-based context implementation."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["LowLevelContext"],
    ):
        super().__init__(context_id, parent_context)

        self.text_chunks: List[TextChunk] = []

    # override
    def destruction(self):
        return super().destruction()

    # override
    def get_this_context_len(self) -> int:
        # This is not useful
        return sum([len(chunk.text) for chunk in self.text_chunks])

    def append_text(self, content: str, role_is_user: bool):
        self.text_chunks.append(
            TextChunk(
                text=content,
                role="user" if role_is_user else "assistant",
            )
        )

    def get_latest_context_text(self) -> str:
        if len(self.text_chunks) == 0:
            return ""
        return self.text_chunks[-1].text

    def get_whole_context_text(self) -> str:
        texts = [text.text for text in self.text_chunks]
        parent_text = (
            ""
            if self.parent_context is None
            else self.parent_context.get_whole_context_text()
        )
        return parent_text + "".join(texts)

    def get_whole_chat_messages(self) -> str:
        messages = [
            {
                "role": text.role,
                "content": text.text,
            }
            for text in self.text_chunks
        ]
        if self.parent_context is not None:
            messages = self.parent_context.get_whole_chat_messages() + messages

        # Merge messages with the same role
        merged_messages = []
        for message in messages:
            if (
                len(merged_messages) > 0
                and merged_messages[-1]["role"] == message["role"]
            ):
                merged_messages[-1]["content"] += message["content"]
            else:
                merged_messages.append(message)
        return merged_messages

    # Text Context doesn't implement the following methods.

    # override
    def get_last_token_id(self) -> int:
        raise NotImplementedError

    # override
    def push_token_id(self, token_id: int):
        raise NotImplementedError
