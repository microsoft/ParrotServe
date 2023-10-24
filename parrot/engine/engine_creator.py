import json

from parrot.utils import get_logger

from .llm_engine import LLMEngine
from .config import (
    ENGINE_TYPE_NATIVE,
    ENGINE_TYPE_HUGGINGFACE,
    ENGINE_TYPE_OPENAI,
    ENGINE_TYPE_MLCLLM,
    EngineConfig,
)
from .native.native_engine import NativeEngine
from .mlc_llm.mlc_engine import MLCEngine

logger = get_logger("Engine Creator")


def create_engine(engine_config_path: str, connect_to_os: bool = True) -> LLMEngine:
    """Create an execution engine.

    Args:
        engine_config_path: str. The path to the engine config file.
        connect_to_os: bool. Whether to connect to the OS.

    Returns:
        LLMEngine. The created execution engine.
    """

    with open(engine_config_path) as f:
        engine_config = dict(json.load(f))

    if not EngineConfig.verify_config(engine_config):
        raise ValueError(f"Invalid engine config: {engine_config}")

    engine_type = engine_config["engine_type"]

    if engine_type == ENGINE_TYPE_NATIVE:
        return NativeEngine(engine_config, connect_to_os)
    elif engine_type == ENGINE_TYPE_MLCLLM:
        return MLCEngine(engine_config, connect_to_os)
    else:
        raise NotImplementedError(f"Unsupported engine type: {engine_type}")
