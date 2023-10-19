from parrot.engine.runtime_info import EngineRuntimeInfo
from parrot.engine.config import EngineConfig, EngineType

from .process.interpret_type import InterpretType


INTERPRET_TYPE_MAP = {
    EngineType.NATIVE: InterpretType.TOKEN_ID,
    EngineType.HUGGINGFACE: InterpretType.TOKEN_ID,
    EngineType.OPENAI: InterpretType.TEXT,
    EngineType.MLCLLM: InterpretType.TEXT,
}


class ExecutionEngine:
    """Represent execution engines in os-level management."""

    def __init__(
        self,
        engine_id: int,
        name: str,
        config: EngineConfig,
    ):
        # ---------- Basic Config ----------
        self.engine_id = engine_id
        self.name = name
        self.config = config

        # ---------- Runtime Info ----------
        self.runtime_info = EngineRuntimeInfo()

    @property
    def http_address(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def interpreter_type(self) -> InterpretType:
        return INTERPRET_TYPE_MAP[self.config.engine_type]
