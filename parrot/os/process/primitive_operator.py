# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List

from parrot.constants import PIPELINE_SEND_CHUNK_NUM, DETOKENIZE_CHUNK_NUM
from parrot.protocol.sampling_config import SamplingConfig

from .placeholder import SVPlaceholder, TokensHolder
from .pipe import TokenPipe


class PrimitiveOperator:
    """Primitive operators are the first stage of a semantic call in the executor.

    When a semantic function is executed, its body will be transformed into a sequence of operators
    by the interpreter. And these operators are then transformed/merged into primitives.

    The different between operators and primitives is that operators are presented as a DAG with
    input holder and output holder, while primitives are pure requests.
    """


class TokenIdConstantFill(PrimitiveOperator):
    """ConstantFill operator takes a constant as input."""

    def __init__(self, token_ids: List[int]):
        super().__init__()
        self.token_ids = token_ids

    def __str__(self) -> str:
        return f"ConstantFill"


class TokenIdPlaceholderFill(PrimitiveOperator):
    """TokenIdPlaceholderFill operator takes an Dataholder as input."""

    def __init__(self, input_holder: TokensHolder):
        super().__init__()
        self.input_holder: TokensHolder = input_holder
        self.input_holder.consumers.append(self)
        self.input_pipe = TokenPipe(PIPELINE_SEND_CHUNK_NUM)

    def __str__(self) -> str:
        return f"TokenIdPlaceholderFill: input={self.input_holder}"


class TokenIdPlaceholderGenerate(PrimitiveOperator):
    """TokenIdPlaceholderGenerate operator takes a Dataholder as an output.
    And the decoded result will be passed back from the backend token by token (streaming).
    """

    def __init__(self, output_holder: TokensHolder, sampling_config: SamplingConfig):
        super().__init__()
        self.output_holder: TokensHolder = output_holder
        self.sampling_config = sampling_config
        assert self.output_holder.producer is None, "Concurrent writing to a holder"
        self.output_holder.producer = self
        self.detokenize_pipe = TokenPipe(DETOKENIZE_CHUNK_NUM)

    def __str__(self) -> str:
        return f"TokenIdPlaceholderGenerate: output={self.output_holder}"


class TextConstantFill(PrimitiveOperator):
    """TextConstantFill operator takes a constant as input.."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __str__(self) -> str:
        return f"TextConstantFill"


class TextPlaceholderFill(PrimitiveOperator):
    """TextPlaceholderFill operator takes a Placeholder as input.."""

    def __init__(self, input_holder: SVPlaceholder):
        super().__init__()
        self.input_holder = input_holder

    def __str__(self) -> str:
        return f"TextPlaceholderFill"


class TextPlaceholderGenerate(PrimitiveOperator):
    """TextPlaceholderGenerate operator takes a Placeholder as output."""

    def __init__(self, output_holder: SVPlaceholder, sampling_config: SamplingConfig):
        super().__init__()
        self.output_holder = output_holder
        self.sampling_config = sampling_config

    def __str__(self) -> str:
        return f"TextPlaceholderGenerate"
