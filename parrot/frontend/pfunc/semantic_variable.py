# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass
from enum import Enum
from typing import Optional

from parrot.sampling_config import SamplingConfig
from parrot.serve.scheduler.schedule_annotation import ScheduleAnnotation


from typing import Optional


class PFuncSemanticVariable:
    """Maintain an object that represents a semantic variable in PFunc frontend."""

    _virtual_machine_env: Optional["VirtualMachine"] = None

    def __init__(
        self,
        name: Optional[str] = None,
        content: Optional[str] = None,
    ):
        self.name = name if name is not None else f"v{self.id}"
        self.content = content

    def __repr__(self) -> str:
        if self.ready:
            return f"Future(name={self.name}, id={self.id}, content={self.content})"
        return f"Future(name={self.name}, id={self.id})"

    # ---------- Public Methods ----------

    @property
    def ready(self) -> bool:
        return self.content is not None

    def set(self, content):
        """Public API: Set the content of the future."""

        self.content = content
        self._virtual_machine_env.placeholder_set_handler(self.id, content)
        return

    def get(self) -> str:
        """Public API: (Blocking) Get the content of the future."""

        if self.ready:
            return self.content
        content = self._virtual_machine_env.placeholder_fetch_handler(self.id)
        return content

    async def aget(self) -> str:
        """Public API: (Asynchronous) Get the content of the future."""

        if self.ready:
            return self.content
        content = await self._virtual_machine_env.aplaceholder_fetch_handler(self.id)
        return content


@dataclass
class SemanticRegion:
    idx: int


@dataclass
class Constant(SemanticRegion):
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
    dispatch_annotation: Optional[ScheduleAnnotation] = None

    @property
    def is_input_loc(self) -> bool:
        return self.typ == ParamType.INPUT_LOC

    @property
    def is_output(self) -> bool:
        return self.typ == ParamType.OUTPUT_LOC


@dataclass
class ParameterLoc(SemanticRegion):
    """An input/output location in the function."""

    param: Parameter
