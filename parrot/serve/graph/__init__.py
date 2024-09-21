# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

"""
Intermediate representation of LLM requests (ChunkedRequest, Graph, Semantic Variable) 
in Parrot OS.
"""

from .call_request import ChunkedSemanticCallRequest, PyNativeCallRequest
from .perf_criteria import PerformanceCriteria, get_performance_criteria
from .semantic_variable import SemanticVariable
from .nodes import (
    SemanticNode,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    NativeFuncNode,
)
from .graph import CompletionChain, RequestChain, ComputeGraph
from .graph_traverse import activate_producer
