from typing import List

from .tokens_holder import TokensHolder
from .pipe import TokenPipe
from ..constants import PIPELINE_SEND_CHUNK_NUM, DETOKENIZE_CHUNK_NUM


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
    """PlaceholderFill instruction takes a placeholder as input."""

    def __init__(self, input_holder: TokensHolder):
        super().__init__()
        self.input_holder: TokensHolder = input_holder
        self.input_holder.consumers.append(self)
        self.input_pipe = TokenPipe(PIPELINE_SEND_CHUNK_NUM)

    def __str__(self) -> str:
        return f"PlaceholderFill: input={self.input_holder}"


class Generation(Instruction):
    """Generation instruction, corresponding to Generation primitive in backend."""

    def __init__(self, output_holder: TokensHolder):
        super().__init__()
        self.output_holder: TokensHolder = output_holder
        assert self.output_holder.producer is None, "Concurrent writing to a holder"
        self.output_holder.producer = self
        self.detokenize_pipe = TokenPipe(DETOKENIZE_CHUNK_NUM)

    def __str__(self) -> str:
        return f"Generation: output={self.output_holder}"
