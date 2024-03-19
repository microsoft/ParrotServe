# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from .model_type import ModelType


class LanguageModel:
    """Represent a large language model in the backend."""

    def __init__(self, model_name: str, model_type: ModelType) -> None:
        self.model_name = model_name
        self.model_type = model_type
