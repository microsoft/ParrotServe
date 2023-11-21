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
    def remain_batch_size(self) -> int:
        return self.config.max_batch_size - self.runtime_info.num_running_jobs
