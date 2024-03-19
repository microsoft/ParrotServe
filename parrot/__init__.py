# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Parrot: Efficient Serving LLM-based Agents with Dependent Semantic Variables.

(Scalable and Efficient Runtime System for Semantic Programming.)
"""

# Program Interface and Transforms
from .frontend.pfunc import *

# Sampling config and annotations
from .serve.scheduler.annotation import DispatchAnnotation
from .sampling_config import SamplingConfig
