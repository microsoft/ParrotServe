from typing import List
from enum import Enum, auto

from parrot.constants import PIPELINE_SEND_CHUNK_NUM, DETOKENIZE_CHUNK_NUM
from parrot.protocol.sampling_config import SamplingConfig
from parrot.program.future import Future

from .dataholder import DataHolder
from .pipe import TokenPipe


class InterpretType(Enum):
    TOKEN_ID = auto()
    TEXT = auto()


class Instruction:
    """Instructions are the first stage of a semantic call in the executor.

    When a semantic function is executed, its body will be transformed into a sequence of instructions
    by the interpreter. And these instructions are then transformed/merged into primitives.

    The different between instructions and primitives is that instructions are presented as a DAG with
    input holder and output holder, while primitives are pure requests.
    """


class TokenIdConstantFill(Instruction):
    """ConstantFill instruction takes a constant as input."""

    def __init__(self, token_ids: List[int]):
        super().__init__()
        self.token_ids = token_ids

    def __str__(self) -> str:
        return f"ConstantFill"


class TokenIdPlaceholderFill(Instruction):
    """TokenIdPlaceholderFill instruction takes an Dataholder as input."""

    def __init__(self, input_holder: DataHolder):
        super().__init__()
        self.input_holder: DataHolder = input_holder
        self.input_holder.consumers.append(self)
        self.input_pipe = TokenPipe(PIPELINE_SEND_CHUNK_NUM)

    def __str__(self) -> str:
        return f"TokenIdPlaceholderFill: input={self.input_holder}"


class TokenIdPlaceholderGenerate(Instruction):
    """TokenIdPlaceholderGenerate instruction takes a Dataholder as an output.
    And the decoded result will be passed back from the backend token by token (streaming).
    """

    def __init__(self, output_holder: DataHolder, sampling_config: SamplingConfig):
        super().__init__()
        self.output_holder: DataHolder = output_holder
        self.sampling_config = sampling_config
        assert self.output_holder.producer is None, "Concurrent writing to a holder"
        self.output_holder.producer = self
        self.detokenize_pipe = TokenPipe(DETOKENIZE_CHUNK_NUM)

    def __str__(self) -> str:
        return f"TokenIdPlaceholderGenerate: output={self.output_holder}"


class TextConstantFill(Instruction):
    """TextConstantFill instruction takes a constant as input.."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __str__(self) -> str:
        return f"TextConstantFill"


class TextPlaceholderFill(Instruction):
    """TextPlaceholderFill instruction takes a Future as input.."""

    def __init__(self, input_holder: Future):
        super().__init__()
        self.input_holder = input_holder

    def __str__(self) -> str:
        return f"TextPlaceholderFill"


class TextPlaceholderGenerate(Instruction):
    """TextPlaceholderGenerate instruction takes a Future as output."""

    def __init__(self, output_holder: Future, sampling_config: SamplingConfig):
        super().__init__()
        self.output_holder = output_holder
        self.sampling_config = sampling_config

    def __str__(self) -> str:
        return f"TextPlaceholderGenerate"
