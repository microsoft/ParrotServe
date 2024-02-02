# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Union, Dict, Optional
from enum import Enum

from parrot.protocol.sampling_config import SamplingConfig


@dataclass
class RequestPlaceholder:
    """A placeholder in the request."""

    name: str
    is_output: bool
    value_type: Optional[str] = None
    var_id: Optional[str] = None
    const_value: Optional[str] = None
    sampling_config: Optional[Union[Dict, SamplingConfig]] = None

    def __post_init__(self):
        # Cast sampling_config to SamplingConfig.
        if self.sampling_config is not None:
            self.sampling_config = SamplingConfig(**self.sampling_config)

        # Check input/output arguments.
        if self.is_output:
            if self.value_type is not None:
                raise ValueError("Output placeholder should not have value_type.")
            if self.const_value is not None:
                raise ValueError("Output placeholder should not have const_value.")
            if self.var_id is not None:
                raise ValueError("Output placeholder should not have var_id.")

            # Default sampling_config for output placeholder.
            if self.sampling_config is None:
                self.sampling_config = SamplingConfig()
        else:
            if self.value_type is None:
                raise ValueError("Input placeholder should have value_type.")

            if self.sampling_config is not None:
                raise ValueError("Input placeholder should not have sampling_config.")

            if self.value_type == "constant":
                if self.const_value is None:
                    raise ValueError(
                        "Input placeholder with value_type=constant should have const_value."
                    )
                if self.var_id is not None:
                    raise ValueError(
                        "Input placeholder with value_type=constant should not have var_id."
                    )
            elif self.value_type == "variable":
                if self.var_id is None:
                    raise ValueError(
                        "Input placeholder with value_type=variable should have var_id."
                    )
                if self.const_value is not None:
                    raise ValueError(
                        "Input placeholder with value_type=variable should not have const_value."
                    )
            else:
                raise ValueError(f"Unknown value_type: {self.value_type}")

    @property
    def should_create(self):
        """Return whether we should created a new SV for this placeholder."""

        return self.is_output or (self.var_id is None and self.const_value is None)
