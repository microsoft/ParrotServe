# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from typing import Dict

from .thread import Thread

from parrot.program.semantic_variable import Constant, ParameterLoc
from parrot.exceptions import parrot_assert

from ..tokenizer import Tokenizer
from .placeholder import SVPlaceholder, TokensHolder
from .primitive_operator import *


class BaseInterpreter(ABC):
    """Abstract base class for all interpreters.

    The interpreter is responsible for interpreting the program, i.e. transforming the
    body of semantic function to instructions, then to Fill/Generate primitives.
    """

    @abstractmethod
    def interpret(self, thread: Thread):
        """Interpret the semantic function body to Fill/Generate primitives.

        Args:
            thread: The thread to interpret. The primitives will be added to the thread queue.
        """


class TokenIdInterpreter(BaseInterpreter):
    """Interpreter based on token ids inputs.

    Func calls with the same tokenizer will be interpreted by the same TokenIdInterpreter. And
    this enables the communication between these func calls.
    """

    def __init__(
        self,
        tokenizer_name: str,
        tokenizer: Tokenizer,
    ):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.tokensholder_map: Dict[int, TokensHolder] = {}

    def interpret(self, thread: Thread):
        tokenized = self.tokenizer.tokenize_func_body(
            thread.call.func,
            self.tokenizer_name,
        )

        eos_token_id = self.tokenizer.get_tokenizer(self.tokenizer_name).eos_token_id

        parrot_assert(len(tokenized) == len(thread.call.func.body), f"Length mismatch: {len(tokenized)} != {len(thread.call.func.body)}")

        # Translate function body to instructions
        for i, piece in enumerate(thread.call.func.body):
            if isinstance(piece, Constant):
                inst = TokenIdConstantFill(tokenized[i])
            elif isinstance(piece, ParameterLoc):
                parrot_assert(
                    piece.param.name in thread.call.bindings,
                    "Param should be assigned.",
                )
                param_value = thread.call.bindings[piece.param.name]

                if isinstance(param_value, str):
                    # Str input or Pyobj input
                    inst = TokenIdConstantFill(
                        self.tokenizer.tokenize(
                            param_value,
                            self.tokenizer_name,
                        )
                    )
                else:
                    parrot_assert(
                        isinstance(param_value, SVPlaceholder),
                        "If not str, must be a placeholder",
                    )
                    holder = self._get_dataholder(param_value)
                    if piece.param.is_output:
                        sampling_config = piece.param.sampling_config
                        # If not ignore_tokenizer_eos, we should add eos_token_id to stop_token_ids
                        if not sampling_config.ignore_tokenizer_eos:
                            sampling_config.stop_token_ids.append(eos_token_id)
                        inst = TokenIdPlaceholderGenerate(
                            output_holder=holder,
                            sampling_config=sampling_config,
                        )
                    else:
                        inst = TokenIdPlaceholderFill(input_holder=holder)
            thread.operators.put_nowait(inst)

    def _get_dataholder(self, placeholder: SVPlaceholder) -> TokensHolder:
        # Create a new data placeholder if not exists
        # Hence, the name of the placeholder must be unique.
        if placeholder.id not in self.tokensholder_map:
            self.tokensholder_map[placeholder.id] = TokensHolder(
                tokenizer_name=self.tokenizer_name,
                tokenizer=self.tokenizer,
                placeholder=placeholder,
            )
        return self.tokensholder_map[placeholder.id]


class TextInterpreter(BaseInterpreter):
    """TextInterpreter will not tokenize the function body. It transforms the
    body into Text instructions."""

    def interpret(self, thread: Thread):
        # Translate function body to instructions
        for piece in thread.call.func.body:
            if isinstance(piece, Constant):
                inst = TextConstantFill(piece.text)
            elif isinstance(piece, ParameterLoc):
                parrot_assert(
                    piece.param.name in thread.call.bindings,
                    "Param should be assigned.",
                )
                param_value = thread.call.bindings[piece.param.name]

                if isinstance(param_value, str):
                    inst = TextConstantFill(param_value)
                else:
                    parrot_assert(
                        isinstance(param_value, SVPlaceholder), "Must be a placeholder"
                    )
                    if piece.param.is_output:
                        sampling_config = piece.param.sampling_config
                        inst = TextPlaceholderGenerate(
                            output_holder=param_value,
                            sampling_config=sampling_config,
                        )
                    else:
                        inst = TextPlaceholderFill(input_holder=param_value)
            thread.operators.put_nowait(inst)
