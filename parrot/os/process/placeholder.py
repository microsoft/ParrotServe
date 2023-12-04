# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional
from asyncio import Event

from parrot.program.function import SemanticCall

from .dag_edge import DAGEdge
from ..tokenizer import Tokenizer


class SVPlaceholder:
    """Placeholder corresponds to a Future (Input/Output semantic variable) in the program.

    In the data-dependency graph, it serves a middle node which holds the data and
    connects the producer and consumers.
    """

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
        self.content = None
        self.max_length = 0

        # Events
        self.start_event: Event = Event()
        self.ready_event: Event = Event()

        # TokenHolders
        self.token_holders: List[TokensHolder] = []

        # DAG
        # An edge is an in-edge if this SV is the out-node of the edge.
        # An edge is an out-edge if this SV is the in-node of the edge.
        self.in_edges: List[DAGEdge] = []
        self.out_edges: List[DAGEdge] = []

    def __repr__(self) -> str:
        if self.ready:
            return (
                f"Placeholder(name={self.name}, id={self.id}, content={self.content})"
            )
        return f"Placeholder(name={self.name}, id={self.id})"

    def set(self, content: str):
        """Set the content of the placeholder."""

        assert self.content is None, "This placeholder is filled"
        self.content = content
        self.ready_event.set()
        self.start_event.set()  # Must started

        # Sync results to token holders
        for token_holder in self.token_holders:
            token_holder.sync_from_placeholder()

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
        placeholder: SVPlaceholder,
    ):
        # ---------- Basic info ----------
        self.token_ids: Optional[List[int]] = None
        self.tokenizer = tokenizer
        self.tokenizer_name: str = tokenizer_name
        self.placeholder = placeholder
        placeholder.token_holders.append(self)

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

    def decode_one_token(
        self,
        prev_output_tokens: List[str],
        new_token_id: int,
        skip_special_tokens: bool,
    ) -> str:
        """Decodes one token and updates prev_output_tokens."""
        new_token, output_text = self.tokenizer.detokenize_incrementally(
            self.tokenizer_name,
            prev_output_tokens,
            new_token_id,
            skip_special_tokens,
        )
        if new_token is not None:
            prev_output_tokens.append(new_token)
        self.placeholder.content = output_text

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
