# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Dict

from parrot.protocol.runtime_info import EngineRuntimeInfo
from parrot.program.semantic_variable import ParameterLoc, Constant

# WARN(chaofan): Import from engine package
from parrot.engine.config import EngineConfig

from parrot.constants import (
    ENGINE_TYPE_BUILTIN,
    ENGINE_TYPE_OPENAI,
    ENGINE_TYPE_MLCLLM,
)
from parrot.exceptions import parrot_assert


from .process.placeholder import SVPlaceholder
from .tokenizer import Tokenizer
from .process.interpret_type import InterpretType


INTERPRET_TYPE_MAP = {
    ENGINE_TYPE_BUILTIN: InterpretType.TOKEN_ID,
    ENGINE_TYPE_OPENAI: InterpretType.TEXT,
    ENGINE_TYPE_MLCLLM: InterpretType.TEXT,
}


class ExecutionEngine:
    """Represent execution engines in os-level management."""

    def __init__(
        self,
        engine_id: int,
        config: EngineConfig,
        tokenizer: Tokenizer,
    ):
        # ---------- Basic Config ----------
        self.engine_id = engine_id
        self.config = config
        self.tokenizer = tokenizer
        self.dead = False  # Mark if the engine is dead

        # ---------- Runtime Info ----------
        self.runtime_info = EngineRuntimeInfo()

        # We maintain a instant list of threads in this self.
        self.threads: List["Thread"] = []
        self.threads_len: Dict[int, int] = {}  # thread uid -> thread length

        # We maintain a instant number of tokens in this self.
        self.tokens_num = 0

    @property
    def name(self) -> str:
        return self.config.engine_name

    @property
    def http_address(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def interpreter_type(self) -> InterpretType:
        return INTERPRET_TYPE_MAP[self.config.engine_type]

    @property
    def remain_thread_locs(self) -> int:
        return self.config.threads_capacity - self.num_threads

    @property
    def num_threads(self) -> int:
        return len(self.threads)

    @property
    def requests_num_upperbound(self) -> int:
        """Return the upperbound of the number of jobs that can be dispatched to this self."""
        return min(
            [self.config.threads_capacity]
            + [thread.requests_num_upperbound for thread in self.threads]
        )

    def count_thread_token_nums(self, thread: "Thread") -> int:
        if self.config.engine_type != ENGINE_TYPE_BUILTIN:
            return 0  # TODO(chaofan): support other engines

        tokenizer_name = self.config.tokenizer

        length = 0

        # Count the length of the thread.
        for region in thread.call.func.body:
            if isinstance(region, ParameterLoc):
                value = thread.call.bindings[region.param.name]
                if isinstance(value, SVPlaceholder):
                    length += value.max_length
                else:
                    parrot_assert(
                        isinstance(value, str), f"Value must be str. Got {type(value)}"
                    )
                    length += len(self.tokenizer.tokenize(value, tokenizer_name))
            else:
                parrot_assert(
                    isinstance(region, Constant),
                    f"Invalid region type: {type(region)}.",
                )
                length += len(self.tokenizer.tokenize(region.text, tokenizer_name))

        return length

    def accept_thread(self, thread: "Thread"):
        """Accept a thread to this self."""

        thread.engine = self
        thread_len = self.count_thread_token_nums(thread)
        self.threads.append(thread)
        self.threads_len[thread.unique_id] = thread_len
        self.tokens_num += thread_len

    def remove_thread(self, thread: "Thread"):
        """Remove a thread from this self."""

        # Don't do this! Because hence the thread will be marked as not dispatched.
        # thread.engine = None

        self.threads.remove(thread)
        self.tokens_num -= self.threads_len[thread.unique_id]
        self.threads_len.pop(thread.unique_id)
