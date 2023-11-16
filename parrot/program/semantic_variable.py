from dataclasses import dataclass
from enum import Enum
from typing import Optional

from parrot.protocol.sampling_config import SamplingConfig


@dataclass
class SemanticVariable:
    """Semantic Variable: the core abstraction in Parrot system.

    Its main purpose is to chunk a LLM request into smaller pieces, so that
    we can do fine-grained management and optimization.

    Definition: a Semantic Variable (SV) is a part of prompts with specific semantic
    purpose. A SV can be:
    - A system prompt of a request (Also called prefix).
    - A user-input of a request (Also called an input / a parameter of a function).
    - An output location of a request (Also called a return value of a function).
    - A communication port of two LLM Agents.
    - A few-shot example of a request.
    - ...

    The motivation is that the prompt itself is structural, and can be split into
    different independent parts with different semantic purposes.
    """

    idx: int


@dataclass
class Constant(SemanticVariable):
    """Constant text."""

    text: str


class ParamType(Enum):
    """Type of a parameter."""

    INPUT_LOC = 0
    OUTPUT_LOC = 1
    INPUT_PYOBJ = 2


@dataclass
class Parameter:
    """Parameter of a function.

    A parameter is a special semantic variable that represents a user-input / output
    of a LLM request.
    """

    name: str
    typ: ParamType
    sampling_config: Optional[SamplingConfig] = None

    @property
    def is_input_loc(self) -> bool:
        return self.typ == ParamType.INPUT_LOC

    @property
    def is_output(self) -> bool:
        return self.typ == ParamType.OUTPUT_LOC


@dataclass
class ParameterLoc(SemanticVariable):
    """An input/output location in the function."""

    param: Parameter
