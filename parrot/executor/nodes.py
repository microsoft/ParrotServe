from typing import List, Optional
from enum import Enum
from asyncio import Event

from ..program.placeholder import Placeholder
from ..orchestration.tokenize import TokenizedStorage


class Tokensholder:
    """Placeholder stores the text while Tokensholder stores the tokenized ids.

    Hence it's tokenizer-related.
    """

    def __init__(
        self,
        placeholder: Optional[Placeholder] = None,
        tokenized_storage: Optional[TokenizedStorage] = None,
    ):
        # ---------- Basic info ----------
        self.token_ids: Optional[List[int]] = None
        self.tokenized_storage = tokenized_storage
        self.placeholder = placeholder
        self.placeholder.assign_callbacks.append(
            self.sync_from_placeholder
        )  # Add callback

        # ---------- Jobs ----------
        self.comsumers: List["FillJob"] = []
        self.producer: Optional["GenerationJob"] = None

        # ---------- Streaming generator ----------
        self.generator: Optional[function] = None

        # ---------- Events ----------
        self.streaming_event: Event = Event()
        self.ready_event: Event = Event()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def assign(self, token_ids: List[int]):
        assert self.ready, "This holder is filled"
        assert self.streaming_event.is_set(), "This holder is streaming"

        self.token_ids = token_ids
        self.ready_event.set()
        # NOTE(chaofan): When it has data, also set the streaming event.
        self.streaming_event.set()

    def sync_from_placeholder(self):
        assert self.placeholder is not None, "No placeholder"
        assert self.placeholder.ready, "Placeholder not ready"
        assert self.tokenized_storage is not None, "No tokenized storage"
        self.assign(self.tokenized_storage.tokenize(self.placeholder.content))

    def sync_to_placeholder(self):
        assert self.placeholder is not None, "No placeholder"
        assert self.ready, "Tokensholder not ready"
        assert self.tokenized_storage is not None, "No tokenized storage"
        self.placeholder.assign(self.tokenized_storage.detokenize(self.token_ids))


class JobStatus(Enum):
    WAITING = 1
    READY = 2
    RUNNING = 3


class Job:
    def __init__(self):
        self.status: JobStatus = JobStatus.WAITING


class FillJob(Job):
    def __init__(self, input_holder: Tokensholder):
        super().__init__()
        self.input_holder: Tokensholder = input_holder
        self.input_holder.comsumers.append(self)


class GenerationJob(Job):
    def __init__(self, output_holder: Tokensholder):
        super().__init__()
        self.output_holder: Tokensholder = output_holder
        assert self.output_holder.producer is None, "Concurrent writing to a holder"
        self.output_holder.producer = self
