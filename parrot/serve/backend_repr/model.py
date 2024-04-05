# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict

from parrot.engine.config import EngineConfig

from parrot.exceptions import ParrotCoreUserError
from parrot.constants import (
    ENGINE_TYPE_BUILTIN,
    ENGINE_TYPE_OPENAI,
)


class ModelType(Enum):
    """Two types of models: TokenId and Text.

    These are also two types of runtime representation, using TokenId or Text as the basic unit
    of communication.
    """

    TOKEN_ID = auto()
    TEXT = auto()


MODEL_TYPE_MAP = {
    ENGINE_TYPE_BUILTIN: ModelType.TOKEN_ID,
    ENGINE_TYPE_OPENAI: ModelType.TEXT,
}


def get_model_type(model_type_str: str) -> ModelType:
    """Get ModelType from a string."""

    if model_type_str == "token_id":
        return ModelType.TOKEN_ID
    elif model_type_str == "text":
        return ModelType.TEXT
    else:
        raise ParrotCoreUserError(ValueError(f"Unknown model type: {model_type_str}"))


@dataclass
class LanguageModel:
    """Represent a large language model in the backend."""

    model_name: str
    tokenizer_name: str
    model_type: ModelType

    @classmethod
    def from_engine_config(cls, engine_config: EngineConfig) -> "LanguageModel":
        """Fetch the model info from an engine config."""

        model_name = engine_config.model
        tokenizer_name = engine_config.tokenizer
        model_type = MODEL_TYPE_MAP[engine_config.engine_type]

        model = cls(model_name, tokenizer_name, model_type)

        return model
