# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List

from parrot_vllm_oldscheduler.protocol.runtime_info import EngineRuntimeInfo
from parrot_vllm_oldscheduler.engine.config import (
    EngineConfig,
    ENGINE_TYPE_BUILTIN,
    ENGINE_TYPE_OPENAI,
    ENGINE_TYPE_MLCLLM,
)

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
    ):
        # ---------- Basic Config ----------
        self.engine_id = engine_id
        self.config = config
        self.dead = False  # Mark if the engine is dead

        # ---------- Runtime Info ----------
        self.runtime_info = EngineRuntimeInfo()
        self.threads: List["Thread"] = []

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
        return self.config.max_threads_num - self.num_threads

    @property
    def num_threads(self) -> int:
        return len(self.threads)

    @property
    def requests_num_upperbound(self) -> int:
        """Return the upperbound of the number of jobs that can be dispatched to this engine."""
        return min(
            [self.config.max_threads_num]
            + [thread.requests_num_upperbound for thread in self.threads]
        )

    def accept_thread(self, thread: "Thread"):
        """Accept a thread to this engine."""

        thread.engine = self
        self.threads.append(thread)

    def remove_thread(self, thread: "Thread"):
        """Remove a thread from this engine."""

        # Don't do this! Because hence the thread will be marked as not dispatched.
        # thread.engine = None

        self.threads.remove(thread)
