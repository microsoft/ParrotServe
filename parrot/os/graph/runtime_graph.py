# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional
from asyncio import Event

from parrot.constants import DETOKENIZE_CHUNK_NUM
from parrot.protocol.sampling_config import SamplingConfig

from .graph import BaseNode
from ..tokenizer import Tokenizer


class RuntimeBaseNode:
    """RuntimeBaseNode is the runtime representation of a BaseNode.

    When a semantic function is executed, its body will be transformed into a sequence of RuntimeNodes
    by the interpreter.
    """

    def __init__(self, node: BaseNode):
        self.node = node


class TokenIdConstantFill(RuntimeBaseNode):
    """TokenIdConstantFill node takes constant token ids as input."""

    def __init__(self, node: BaseNode, token_ids: List[int]):
        super().__init__(node)
        self.token_ids = token_ids

    def __str__(self) -> str:
        return f"TokenIdConstantFill"


class TokenIdPlaceholderFill(RuntimeBaseNode):
    """TokenIdPlaceholderFill node takes an Dataholder as input."""

    def __init__(self, node: BaseNode, input_holder: TokensIdPlaceholder):
        super().__init__(node)
        self.input_holder: TokensIdPlaceholder = input_holder

    def __str__(self) -> str:
        return f"TokenIdPlaceholderFill: input={self.input_holder}"


class TokenIdPlaceholderGenerate(RuntimeBaseNode):
    """TokenIdPlaceholderGenerate operator takes a Dataholder as an output.
    And the decoded result will be passed back from the backend token by token (streaming).
    """

    def __init__(
        self,
        node: BaseNode,
        output_holder: TokensIdPlaceholder,
        sampling_config: SamplingConfig,
    ):
        super().__init__(node)
        self.output_holder: TokensIdPlaceholder = output_holder
        self.sampling_config = sampling_config

    def __str__(self) -> str:
        return f"TokenIdPlaceholderGenerate: output={self.output_holder}"


class TextConstantFill(RuntimeBaseNode):
    """TextConstantFill operator takes a constant as input.."""

    def __init__(self, node: BaseNode, text: str):
        super().__init__(node)
        self.text = text

    def __str__(self) -> str:
        return f"TextConstantFill"


class TextPlaceholderFill(RuntimeBaseNode):
    """TextPlaceholderFill operator takes a Placeholder as input.."""

    def __init__(self, node: BaseNode, input_holder: TextPlaceholder):
        super().__init__(node)
        self.input_holder = input_holder

    def __str__(self) -> str:
        return f"TextPlaceholderFill"


class TextPlaceholderGenerate(RuntimeBaseNode):
    """TextPlaceholderGenerate operator takes a Placeholder as output."""

    def __init__(
        self,
        node: BaseNode,
        output_holder: TextPlaceholder,
        sampling_config: SamplingConfig,
    ):
        super().__init__(node)
        self.output_holder = output_holder
        self.sampling_config = sampling_config

    def __str__(self) -> str:
        return f"TextPlaceholderGenerate"
