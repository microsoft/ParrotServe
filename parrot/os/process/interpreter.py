from abc import ABC, abstractmethod
from typing import Dict

from .thread import Thread

from parrot.program.function import Constant, ParameterLoc, ParamType
from parrot.program.future import Future
from parrot.utils import create_task_in_loop

from .tokenizer import Tokenizer
from .placeholder import DataHolder
from ...protocol.primitives.operator import *


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
        self.dataholder_map: Dict[int, DataHolder] = {}

    def interpret(self, thread: Thread):
        tokenized = self.tokenizer.tokenize_func_body(
            thread.call.func,
            self.tokenizer_name,
        )

        eos_token_id = self.tokenizer.get_tokenizer(self.tokenizer_name).eos_token_id

        # Translate function body to instructions
        for i, piece in enumerate(thread.call.func.body):
            if isinstance(piece, Constant):
                inst = TokenIdConstantFill(tokenized[i])
            elif isinstance(piece, ParameterLoc):
                assert piece.param.name in thread.call.bindings
                param_value = thread.call.bindings[piece.param.name]

                if piece.param.typ == ParamType.PYOBJ:
                    # For Python object, we directly fill the value.
                    # We use __str__ instead of __repr__
                    value_str = str(param_value)
                    inst = TokenIdConstantFill(
                        self.tokenizer.tokenize(
                            value_str,
                            self.tokenizer_name,
                        )
                    )
                else:
                    assert isinstance(param_value, Future)
                    holder = self._get_dataholder(param_value)
                    if piece.param.is_output:
                        assert param_value.is_middle_node
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
            thread.instructions.put_nowait(inst)

        create_task_in_loop(thread.executing())

    def _get_dataholder(self, future: Future) -> DataHolder:
        # Create a new data future if not exists
        # Hence, the name of the future must be unique.
        if future.id not in self.dataholder_map:
            self.dataholder_map[future.id] = DataHolder(
                tokenizer=self.tokenizer_name,
                tokenizer=self.tokenizer,
                future=future,
            )
        return self.dataholder_map[future.id]


class TextInterpreter(BaseInterpreter):
    """TextInterpreter will not tokenize the function body. It transforms the
    body into Text instructions."""

    def interpret(self, thread: Thread):
        # Translate function body to instructions
        for piece in thread.call.func.body:
            if isinstance(piece, Constant):
                inst = TextConstantFill(piece.text)
            elif isinstance(piece, ParameterLoc):
                assert piece.param.name in thread.call.bindings
                param_value = thread.call.bindings[piece.param.name]

                if piece.param.typ == ParamType.PYOBJ:
                    # For Python object, we directly fill the value.
                    # We use __str__ instead of __repr__
                    value_str = str(param_value)
                    inst = TextConstantFill(value_str)
                else:
                    assert isinstance(param_value, Future)
                    if piece.param.is_output:
                        assert param_value.is_middle_node
                        sampling_config = piece.param.sampling_config
                        inst = TextPlaceholderGenerate(
                            output_holder=param_value,
                            sampling_config=sampling_config,
                        )
                    else:
                        inst = TextPlaceholderFill(input_holder=param_value)
            thread.instructions.put_nowait(inst)

        create_task_in_loop(thread.executing())
