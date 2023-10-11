from enum import Enum, auto


class EngineType(Enum):
    NATIVE = auto()
    HUGGINGFACE = auto()
    OPENAI = auto()
    MLCLLM = auto()
