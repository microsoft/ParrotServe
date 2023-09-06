from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from asyncio import Event

from ..program.placeholder import Placeholder


@dataclass
class Tokensholder:
    """Placeholder stores the text while Tokensholder stores the tokenized ids.

    Hence it's tokenizer-related.
    """

    token_ids: Optional[List[int]] = None
    placeholder: Optional[Placeholder] = None
    # Who consume contents in this holder.
    comsumers: List["FillJob"] = []
    # Who produce content into this holder.
    producer: Optional["GenerationJob"] = None
    ready_event: Event = Event()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def assign(self, token_ids: List[int]):
        assert self.token_ids is None, "This holder is filled"
        self.token_ids = token_ids
        self.ready_event.set()


class JobStatus(Enum):
    WAITING = 1
    READY = 2
    RUNNING = 3


@dataclass
class Job:
    status: JobStatus = JobStatus.WAITING


@dataclass
class FillJob(Job):
    input_holder: Tokensholder


@dataclass
class GenerationJob(Job):
    output_holder: Tokensholder
