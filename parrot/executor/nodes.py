from typing import List, Optional
from enum import Enum
from asyncio import Event, Queue

from ..program.placeholder import Placeholder
from ..orchestration.tokenize import TokenizedStorage


class Tokensholder:
    """Placeholder stores the text while Tokensholder stores the tokenized ids.

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

    def sync_to_placeholder(self):
        assert self.placeholder is not None, "No placeholder"
        assert self.ready, "Tokensholder not ready"
        assert self.tokenized_storage is not None, "No tokenized storage"

        # Remove the callback to avoid infinite loop
        # And no need to add back, since the tokenholder is ready
        self.placeholder.assign_callbacks.remove(self.sync_from_placeholder)
        self.placeholder.assign(
            self.tokenized_storage.detokenize(
                self.token_ids,
                self.tokenizer,
            )
        )

    def __str__(self) -> str:
        if self.is_constant:
            return f"[Tokensholder(Constant): {self.token_ids}]"
        return f"[Tokensholder(Placeholder): {self.placeholder.name}]"


class JobStatus(Enum):
    WAITING = 1
    RUNNING = 2


class Job:
    def __init__(self):
        self.status: JobStatus = JobStatus.WAITING


class FillJob(Job):
    """Fill job is corresponding to the `prefill` stage in LLM.

    Its mission is to fill the KV cache in the execution engine, extending the context
    using the input tokens.
    """

    def __init__(self, input_holder: Tokensholder):
        super().__init__()
        self.input_holder: Tokensholder = input_holder
        self.input_holder.consumers.append(self)
        self.pipe: Queue[int] = Queue()

    def __str__(self) -> str:
        return f"FillJob: input={self.input_holder}"


class GenerationJob(Job):
    """Generation job is corresponding to the `decode` stage in LLM.

    Its mission is to generate the output tokens based on certain context.
    """

    def __init__(self, output_holder: Tokensholder):
        super().__init__()
        self.output_holder: Tokensholder = output_holder
        assert self.output_holder.producer is None, "Concurrent writing to a holder"
        self.output_holder.producer = self

    def __str__(self) -> str:
        return f"GenerationJob: output={self.output_holder}"
