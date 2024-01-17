# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum, auto

from parrot.exceptions import ParrotOSUserError


class ModelType(Enum):
    """Two types of models: TokenId and Text.

    These are also two types of runtime representation, using TokenId or Text as the basic unit
    of communication.
    """

    TOKEN_ID = auto()
    TEXT = auto()


def get_model_type(model_type_str: str) -> ModelType:
    """Get ModelType from a string."""

    if model_type_str == "token_id":
        return ModelType.TOKEN_ID
    elif model_type_str == "text":
        return ModelType.TEXT
    else:
        raise ParrotOSUserError(ValueError(f"Unknown model type: {model_type_str}"))
