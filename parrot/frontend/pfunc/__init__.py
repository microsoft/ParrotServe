# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

"""
PFunc is parrot's native frontend, built on top of Parrot's API.
"""

# VirtualMachine
from .vm import VirtualMachine

# Interface
from .interface import Input, Output, semantic_function, native_function, variable

from .function import Parameter, ParamType  # For define functions

# Useful transforms and sequential transforms
from .transforms.prompt_formatter import standard_formatter, allowing_newline
from .transforms.conversation_template import vicuna_template

# Performance criteria
from .perf_criteria import PerformanceCriteria

# SamplingConfig
from parrot.sampling_config import SamplingConfig
