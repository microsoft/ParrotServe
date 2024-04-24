# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""
Semantic Variable: the core abstraction in Parrot system.

Its main purpose is to chunk a LLM request into smaller pieces, so that
we can do fine-grained management and optimization.

Definition: a Semantic Variable (SV) is a part of prompts with specific semantic
purpose. A SV can be:
- 1. A system prompt of a request (Also called prefix).
- 2. A user-input of a request (Also called an input / a parameter of a function).
- 3. An output location of a request (Also called a return value of a function).
- 4. A communication port of two LLM Agents.
- 5. A few-shot example of a request.
- ...

The motivation is that the prompt itself is structural, and can be split into
different independent parts with different semantic purposes.


NOTE(chaofan):

In real code implementation, we use the name "Semantic Variable" to refer specifically 
to the input/output locations in the prompts (which is 2. & 3. in the definition above), 
to align with the APIs in paper.

And we refer to those "part of prompts with specific semantic purpose" as "Semantic Region", 
because they don't flow across requests, which is more "static".

But in a word, it's just a naming issue, not a big deal.
"""


from dataclasses import dataclass
from enum import Enum
from typing import Optional

from parrot.protocol.sampling_config import SamplingConfig
from parrot.protocol.annotation import DispatchAnnotation


from typing import Optional


class SemanticVariable:
    """Represents a string which will be filled in the Future.

    It's like "Future" in the Python asynchronous programming, or "Promise" in JavaScript.
    As its name suggests, it's a placeholder for the content to be filled in the future.

    It also corresponds to a Input/Output semantic variable in Parrot System.
    """

    _counter = 0
    _virtual_machine_env: Optional["VirtualMachine"] = None

    def __init__(
        self,
        name: Optional[str] = None,
        content: Optional[str] = None,
    ):
        self.id = self._increment()
        self.name = name if name is not None else f"v{self.id}"
        self.content = content

    @classmethod
    def _increment(cls) -> int:
        cls._counter += 1
        return cls._counter

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
    dispatch_annotation: Optional[DispatchAnnotation] = None

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
