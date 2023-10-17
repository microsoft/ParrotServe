from parrot.engine.config import EngineConfig, EngineType
from parrot.constants import FILL_NO_CHUNK

from .process.interpreter import InterpretType


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
        name: str,
        host: str,
        port: int,
        config: EngineConfig,
    ):
        # ---------- Basic Config ----------
        self.name = name
        self.host = host
        self.port = port
        self.config = config

        # ---------- Monitor Data ----------
        self.num_cached_tokens = 0
        self.cached_tokens_size = 0
        self.num_running_jobs = 0

        # ---------- Controlled Args ----------
        self.fill_chunk_size = FILL_NO_CHUNK

    @property
    def http_address(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def interpreter_type(self) -> InterpretType:
        return INTERPRET_TYPE_MAP[self.config.engine_type]
