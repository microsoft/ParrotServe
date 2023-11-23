# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from parrot.protocol.engine_runtime_info import EngineRuntimeInfo
from parrot.engine.config import (
    EngineConfig,
    ENGINE_TYPE_NATIVE,
    ENGINE_TYPE_OPENAI,
    ENGINE_TYPE_MLCLLM,
)

from .process.interpret_type import InterpretType


INTERPRET_TYPE_MAP = {
    ENGINE_TYPE_NATIVE: InterpretType.TOKEN_ID,
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
        self.num_threads = 0
        self.expect_max_jobs_num = (
            99999  # = min{thread.max_jobs_num for each thread in this engine}
        )

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
    def jobs_num(self) -> int:
        return self.runtime_info.num_total_jobs

    @property
    def remain_job_locs(self) -> int:
        return self.config.max_jobs_num - self.jobs_num

    def accept_thread(self, thread: "Thread"):
        """Accept a thread to this engine."""

        thread.engine = self
        self.num_threads += 1

    def remove_thread(self, thread: "Thread"):
        """Remove a thread from this engine."""

        thread.engine = None
        self.num_threads -= 1
