# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional
from enum import Enum
from dataclasses import dataclass

from parrot.sampling_config import SamplingConfig


@dataclass
class FuncBodyPiece:
    idx: int


@dataclass
class Constant(FuncBodyPiece):
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

    def get_param_str(self) -> str:
        return "{{" + self.name + "}}"


@dataclass
class ParameterLoc(FuncBodyPiece):
    """An input/output location in the function."""

    param: Parameter
