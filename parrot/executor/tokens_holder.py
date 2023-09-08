from typing import List, Optional
from asyncio import Event

from ..program.placeholder import Placeholder
from ..orchestration.tokenize import TokenizedStorage


class TokensHolder:
    """Placeholder stores the text while TokensHolder stores the tokenized ids.

    Hence it's tokenizer-related.
    """

    def __init__(
        self,
        tokenizer: str,
        tokenized_storage: TokenizedStorage,
        placeholder: Optional[Placeholder] = None,
    ):
        # ---------- Basic info ----------
        self.token_ids: Optional[List[int]] = None
        self.tokenized_storage = tokenized_storage
        self.tokenizer: str = tokenizer
        self.placeholder = placeholder

        # ---------- Jobs ----------
        self.consumers: List["FillJob"] = []
        self.producer: Optional["GenerationJob"] = None

        # ---------- Events ----------
        self.streaming_event: Event = Event()
        self.ready_event: Event = Event()

        if placeholder is not None:
            self.placeholder.assign_callbacks.append(
                self.sync_from_placeholder
            )  # Add callback
            if placeholder.ready:
                self.sync_from_placeholder()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    @property
    def is_constant(self) -> bool:
        return self.placeholder is None

    def assign(self, token_ids: List[int]):
        assert not self.ready, "This tokenholder is filled. Can't assign."
        assert (
            not self.streaming_event.is_set()
        ), "This tokeholder is streaming. Can't assign."

        self.token_ids = token_ids
        self.ready_event.set()
        # NOTE(chaofan): When it has data, also set the streaming event.
        self.streaming_event.set()

    def sync_from_placeholder(self):
        assert self.placeholder is not None, "No placeholder"
        assert self.placeholder.ready, "Placeholder not ready"
        assert self.tokenized_storage is not None, "No tokenized storage"
        self.assign(
            self.tokenized_storage.tokenize(
                self.placeholder.content,
                self.tokenizer,
            )
        )

    def sync_to_placeholder_partial(self, token_ids: List[int], is_last_batch: bool):
        assert self.placeholder is not None, "No placeholder"
        assert self.tokenized_storage is not None, "No tokenized storage"

        if self.placeholder.content is None:
            self.placeholder.content = ""

        self.placeholder.content += self.tokenized_storage.detokenize(
            token_ids,
            self.tokenizer,
        )

        if is_last_batch:
            self.placeholder.ready_event.set()

    def __str__(self) -> str:
        if self.is_constant:
            return f"[TokensHolder(Constant): {self.token_ids}]"
        return f"[TokensHolder(Placeholder): {self.placeholder.name}]"
