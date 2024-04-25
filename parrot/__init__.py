# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Parrot: Efficient Serving LLM-based Agents with Dependent Semantic Variables.

(Scalable and Efficient Runtime System for Semantic Programming.)
"""

try:
    import mlc_chat  # Avoid MLC error because "torch" is imported before "mlc_chat"
except ImportError:
    # print("Warning: MLC is not installed. Related functionalities will be disabled.")
    pass

# Program Interface and Transforms
from .program import *

# Sampling config and annotations
from .protocol.annotation import DispatchAnnotation
from .protocol.sampling_config import SamplingConfig
