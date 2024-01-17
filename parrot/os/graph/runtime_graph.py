# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional
from asyncio import Event

from parrot.constants import DETOKENIZE_CHUNK_NUM
from parrot.protocol.sampling_config import SamplingConfig

from .graph import BaseNode, TextPlaceholder
from ..tokenizer import Tokenizer
from .token_pipe import TokenPipe


# ---------- TokensId Placeholder ----------


class TokensIdPlaceholder:
    """TokensIdPlaceholder stores the tokens of a TextPlaceholder. It's tokenizer-specific."""

    def __init__(
        self,
        tokenizer_name: str,
        tokenizer: Tokenizer,
        placeholder: TextPlaceholder,
    ):
        # ---------- Basic info ----------
        self.token_ids: Optional[List[int]] = None
        self.tokenizer = tokenizer
        self.tokenizer_name: str = tokenizer_name
        self.placeholder = placeholder
        placeholder.tokens_id_holders.append(self)

        # ---------- Events ----------
        self.streaming_event: Event = Event()
        self.ready_event: Event = Event()

        # ---------- Detokenize ----------
        self.detokenize_pipe = TokenPipe(DETOKENIZE_CHUNK_NUM)

        if placeholder.ready:
            self.sync_from_placeholder()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def assign(self, token_ids: List[int]):
        assert not self.ready, "This TokensIdPlaceholder is filled. Can't assign."
        assert (
            not self.streaming_event.is_set()
        ), "This TokensIdPlaceholder is streaming. Can't assign."

        self.token_ids = token_ids
        self.ready_event.set()
        # NOTE(chaofan): When it has data, also set the streaming event.
        self.streaming_event.set()

    def sync_from_placeholder(self):
        assert self.placeholder.ready, "Future not ready"
        assert self.tokenizer is not None, "No tokenizer"
        self.assign(
            self.tokenizer.tokenize(
                self.placeholder.content,
                self.tokenizer_name,
            )
        )

    def sync_to_placeholder_partial(
        self, token_ids: List[int], prev_last_token: Optional[int]
    ):
        if self.placeholder.content is None:
            self.placeholder.content = ""

        if prev_last_token:
            token_ids = [prev_last_token] + token_ids
            prev_last_text = self.tokenizer.detokenize(
                [prev_last_token],
                self.tokenizer_name,
            )

        partial_text = self.tokenizer.detokenize(
            token_ids,
            self.tokenizer_name,
        )

        if prev_last_token:
            partial_text = partial_text[len(prev_last_text) :]

        self.placeholder.content += partial_text

    # TODO(chaofan): Fix detokenize_incrementally
    # def decode_one_token(
    #     self,
    #     prev_output_tokens: List[str],
    #     new_token_id: int,
    #     skip_special_tokens: bool,
    # ) -> str:
    #     """Decodes one token and updates prev_output_tokens."""
    #     new_token, output_text = self.tokenizer.detokenize_incrementally(
    #         self.tokenizer_name,
    #         prev_output_tokens,
    #         new_token_id,
    #         skip_special_tokens,
    #     )
    #     if new_token is not None:
    #         prev_output_tokens.append(new_token)
    #     self.placeholder.content = output_text

    def __str__(self) -> str:
        return f"[TokensIdPlaceholder: name={self.placeholder.name}, id={self.placeholder.id}]"

    def send_token(self, token_id: int, put_into_holder: bool = True):
        assert (
            self.streaming_event.is_set()
        ), "This TokensIdPlaceholder is not streaming."
        assert not self.ready, "This TokensIdPlaceholder is filled. Can't send token."

        # Pushing to output holder
        if put_into_holder:
            self.token_ids.append(token_id)

        # Detokenize
        self.detokenize_pipe.queue.put_nowait(token_id)


# ---------- Graph ----------


class RuntimeBaseNode:
    """RuntimeBaseNode is the runtime representation of a BaseNode.

    When a semantic function is executed, its body will be transformed into a sequence of RuntimeNodes
    by the interpreter. And these nodes are then transformed/merged into primitives.
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
