# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Parrot: Efficient Serving LLM-based Agents with Dependent Semantic Variables.

(Scalable and Efficient Runtime System for Semantic Programming.)
"""

# Program Interface and Transforms
from .frontend import *

# Sampling config and annotations
from .os.sv.annotation import DispatchAnnotation
from .protocol.sampling_config import SamplingConfig
