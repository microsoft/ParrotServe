from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from ..program.placeholder import Placeholder


@dataclass
class Tokensholder:
    """Placeholder stores the text while Tokensholder stores the tokenized ids.

    Hence it's tokenizer-related.
    """

    token_ids: List[int]
    placeholder: Optional[Placeholder] = None
    # Who consume contents in this holder.
    comsumers: List["FillJob"]
    # Who produce content into this holder.
    producer: "GenerationJob"


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
