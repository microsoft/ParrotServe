# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import json
from typing import Dict

from parrot.utils import get_logger

from .llm_engine import LLMEngine
from .config import (
    ENGINE_TYPE_NATIVE,
    ENGINE_TYPE_OPENAI,
    ENGINE_TYPE_MLCLLM,
    EngineConfig,
)
from .native.native_engine import NativeEngine
from .mlc_llm.mlc_engine import MLCEngine
from .openai.openai_engine import OpenAIEngine

from parrot.exceptions import ParrotEngineInteralError

logger = get_logger("Engine Creator")


def create_engine(
    engine_config_path: str,
    connect_to_os: bool = True,
    override_args: Dict = {},
) -> LLMEngine:
    """Create an execution engine.

    Args:
        engine_config_path: str. The path to the engine config file.
        connect_to_os: bool. Whether to connect to the OS.
        override_args: Dict. The override arguments.

    Returns:
        LLMEngine. The created execution engine.
    """

    with open(engine_config_path) as f:
        engine_config = dict(json.load(f))

    if "device" in override_args:
        engine_config["instance"]["device"] = override_args["device"]
        override_args.pop("device")
    engine_config.update(override_args)

    if not EngineConfig.verify_config(engine_config):
        raise ParrotEngineInteralError(f"Invalid engine config: {engine_config}")

    engine_type = engine_config["engine_type"]

    if engine_type == ENGINE_TYPE_NATIVE:
        return NativeEngine(engine_config, connect_to_os)
    elif engine_type == ENGINE_TYPE_MLCLLM:
        return MLCEngine(engine_config, connect_to_os)
    elif engine_type == ENGINE_TYPE_OPENAI:
        return OpenAIEngine(engine_config, connect_to_os)
    else:
        raise ParrotEngineInteralError(f"Unsupported engine type: {engine_type}")
