from typing import List
from dataclasses import dataclass
from ..protocol.sampling_params import SamplingParams


@dataclass
class BackendPrimitives:
    session_id: int
    context_id: int


@dataclass
class Fill(BackendPrimitives):
    parent_context_id: int
    tokens_id: List[int]


@dataclass
class Generation(BackendPrimitives):
    sampling_params: SamplingParams
