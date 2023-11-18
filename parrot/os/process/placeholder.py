# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional
from asyncio import Event

from .dag_node import DAGNode
from ..tokenizer import Tokenizer


class Placeholder:
    """Placeholder corresponds to a Future in the program.

    In the data-dependency graph, it serves a middle node which holds the data and
    connects the producer and consumers.
    """

    def __init__(self, id: int):
        self.id = id
        self.content = None
        self.start_event: Event = Event()
        self.ready_event: Event = Event()
        self.out_nodes: List[DAGNode] = []
        self.assign_callbacks = []

    def __repr__(self) -> str:
        if self.ready:
            return f"Placeholder(id={self.id}, content={self.content})"
        return f"Placeholder(id={self.id})"

    def set(self, content: str):
        """Set the content of the placeholder."""
        assert self.content is None, "This placeholder is filled"
        self.content = content
        self.ready_event.set()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    async def get(self) -> str:
        """Get the content of the placeholder."""
        await self.ready_event.wait()
        return self.content


class TokensHolder:
    """TokensHolder stores the tokens of a Placeholder."""

    def __init__(
        self,
        tokenizer_name: str,
        tokenizer: Tokenizer,
        placeholder: Placeholder,
    ):
        # ---------- Basic info ----------
        self.token_ids: Optional[List[int]] = None
        self.tokenizer = tokenizer
        self.tokenizer_name: str = tokenizer_name
        self.placeholder = placeholder

        # ---------- Operators ----------
        self.consumers: List["TokenIdPlaceholderFill"] = []
        self.producer: Optional["TokenIdPlaceholderGenerate"] = None

        # ---------- Events ----------
        self.streaming_event: Event = Event()
        self.ready_event: Event = Event()

        if placeholder.ready:
            self.sync_from_placeholder()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def assign(self, token_ids: List[int]):
        assert not self.ready, "This DataHolder is filled. Can't assign."
        assert (
            not self.streaming_event.is_set()
        ), "This DataHolder is streaming. Can't assign."

        self.token_ids = token_ids
        self.ready_event.set()
        # NOTE(chaofan): When it has data, also set the streaming event.
        self.streaming_event.set()

    def sync_from_placeholder(self):
        assert self.placeholder.ready, "Future not ready"
        assert self.tokenizer is not None, "No tokenizer"
        self.assign(
            self.tokenizer.tokenize(
                self.placeholder.content,
                self.tokenizer_name,
            )
        )

    def sync_to_placeholder_partial(
        self, token_ids: List[int], prev_last_token: Optional[int]
    ):
        if self.placeholder.content is None:
            self.placeholder.content = ""

        if prev_last_token:
            token_ids = [prev_last_token] + token_ids
            prev_last_text = self.tokenizer.detokenize(
                [prev_last_token],
                self.tokenizer_name,
            )

        partial_text = self.tokenizer.detokenize(
            token_ids,
            self.tokenizer_name,
        )

        if prev_last_token:
            partial_text = partial_text[len(prev_last_text) :]

        self.placeholder.content += partial_text

    def __str__(self) -> str:
        return f"[DataHolder: future_id={self.placeholder.id}]"

    def send_token(self, token_id: int, put_into_holder: bool = True):
        assert self.streaming_event.is_set(), "This DataHolder is not streaming."
        assert not self.ready, "This DataHolder is filled. Can't send token."

        # Pushing to output holder
        if put_into_holder:
            self.token_ids.append(token_id)

        # Routing to pipes
        for consumer in self.consumers:
            consumer.input_pipe.queue.put_nowait(token_id)
        assert self.producer is not None, "No producer"
        self.producer.detokenize_pipe.queue.put_nowait(token_id)
