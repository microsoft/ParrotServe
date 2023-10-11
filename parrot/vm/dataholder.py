from typing import List, Optional
from asyncio import Event

from parrot.program.future import Future
from Parrot.parrot.vm.tokenizer import Tokenizer


class DataHolder:
    """DataHolder stores the tokens of a Future.

    In the data-dependency graph, it serves a middle node which holds the data (token_ids) and
    connects the producer and consumers.
    """

    def __init__(
        self,
        tokenizer: str,
        tokenized_storage: Tokenizer,
        future: Future,
    ):
        # ---------- Basic info ----------
        self.token_ids: Optional[List[int]] = None
        self.tokenized_storage = tokenized_storage
        self.tokenizer: str = tokenizer
        self.future = future

        # ---------- Jobs ----------
        self.consumers: List["FillJob"] = []
        self.producer: Optional["GenerationJob"] = None

        # ---------- Events ----------
        self.streaming_event: Event = Event()
        self.ready_event: Event = Event()

        self.future.assign_callbacks.append(self.sync_from_future)  # Add callback
        if future.ready:
            self.sync_from_future()

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

    def sync_from_future(self):
        # assert self.future is not None, "No future"
        assert self.future.ready, "Future not ready"
        assert self.tokenized_storage is not None, "No tokenized storage"
        self.assign(
            self.tokenized_storage.tokenize(
                self.future.content,
                self.tokenizer,
            )
        )

    def sync_to_future_partial(
        self, token_ids: List[int], prev_last_token: Optional[int]
    ):
        # assert self.future is not None, "No future"
        # assert self.tokenized_storage is not None, "No tokenized storage"

        if self.future.content is None:
            self.future.content = ""

        if prev_last_token:
            token_ids = [prev_last_token] + token_ids
            prev_last_text = self.tokenized_storage.detokenize(
                [prev_last_token],
                self.tokenizer,
            )

        partial_text = self.tokenized_storage.detokenize(
            token_ids,
            self.tokenizer,
        )

        if prev_last_token:
            partial_text = partial_text[len(prev_last_text) :]

        self.future.content += partial_text

    def __str__(self) -> str:
        return f"[DataHolder: future_id={self.future.id}]"

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
