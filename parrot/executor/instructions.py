from typing import List

from parrot.constants import PIPELINE_SEND_CHUNK_NUM, DETOKENIZE_CHUNK_NUM
from parrot.protocol.sampling_config import SamplingConfig

from .dataholder import DataHolder
from .pipe import TokenPipe


class Instruction:
    """An instruction is a part of a semantic function.

    When a semantic function is executed, it will be split into several instructions.
    And these instructions are transformed/merged into serveral primitive in backend.
    """


class ConstantFill(Instruction):
    """ConstantFill instruction takes a constant as input."""

    def __init__(self, token_ids: List[int]):
        super().__init__()
        self.token_ids = token_ids

    def __str__(self) -> str:
        return f"ConstantFill"


class PlaceholderFill(Instruction):
    """PlaceholderFill instruction takes an dataholder as input."""

    def __init__(self, input_holder: DataHolder):
        super().__init__()
        self.input_holder: DataHolder = input_holder
        self.input_holder.consumers.append(self)
        self.input_pipe = TokenPipe(PIPELINE_SEND_CHUNK_NUM)

    def __str__(self) -> str:
        return f"PlaceholderFill: input={self.input_holder}"


class PlaceholderGeneration(Instruction):
    """PlaceholderGeneration instruction, corresponding to Generation primitive in backend.

    It takes a dataholder as an output. And the decoded result will be passed back from the backend
    token by token (streaming).
    """

    def __init__(self, output_holder: DataHolder, sampling_config: SamplingConfig):
        super().__init__()
        self.output_holder: DataHolder = output_holder
        self.sampling_config = sampling_config
        assert self.output_holder.producer is None, "Concurrent writing to a holder"
        self.output_holder.producer = self
        self.detokenize_pipe = TokenPipe(DETOKENIZE_CHUNK_NUM)

    def __str__(self) -> str:
        return f"PlaceholderGeneration: output={self.output_holder}"


class TextFill(Instruction):
    """TextFill instruction."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __str__(self) -> str:
        return f"TextFill"


class TextGeneration(Instruction):
    """TextGeneration instruction."""

    def __init__(self, sampling_config: SamplingConfig):
        super().__init__()
        self.sampling_config = sampling_config

    def __str__(self) -> str:
        return f"TextGeneration"
