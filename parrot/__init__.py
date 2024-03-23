# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Parrot: Efficient Serving LLM-based Applications with Dependent Semantic Variables."""

__version__ = "0.01"

# Program Interface and Transforms
from .frontend.pfunc import *

# Sampling config and annotations
from .serve.graph.annotation import DispatchAnnotation
from .sampling_config import SamplingConfig
