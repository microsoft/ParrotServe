from dataclasses import dataclass

from parrot.engine.config import EngineConfig, EngineType
from parrot.constants import FILL_NO_CHUNK

from .process.interpreter import InterpretType


INTERPRET_TYPE_MAP = {
    EngineType.NATIVE: InterpretType.TOKEN_ID,
    EngineType.HUGGINGFACE: InterpretType.TOKEN_ID,
    EngineType.OPENAI: InterpretType.TEXT,
    EngineType.MLCLLM: InterpretType.TEXT,
}


@dataclass
class EngineRuntimeInfo:
    num_cached_tokens: int = 0
    num_running_jobs: int = 0
    cache_mem: int = 0
    model_mem: int = 0


class ExecutionEngine:
    """Represent execution engines in os-level management."""

    def __init__(
        self,
        engine_id: int,
        name: str,
        host: str,
        port: int,
        config: EngineConfig,
    ):
        # ---------- Basic Config ----------
        self.engine_id = engine_id
        self.name = name
        self.host = host
        self.port = port
        self.config = config

        # ---------- Runtime Info ----------
        self.runtime_info = EngineRuntimeInfo()

        # ---------- Controlled Args ----------
        self.fill_chunk_size = FILL_NO_CHUNK

    @property
    def http_address(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def interpreter_type(self) -> InterpretType:
        return INTERPRET_TYPE_MAP[self.config.engine_type]
