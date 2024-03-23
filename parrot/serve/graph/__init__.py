# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

"""
Intermediate representation of LLM requests (ChunkedRequest, Graph, Semantic Variable) 
in Parrot OS.
"""

from .chunked_request import ChunkedRequest
from .semantic_variable import SemanticVariable
from .nodes import BaseNode, ConstantFill, PlaceholderFill, PlaceholderGen
from .node_struct import CompletionChain, RequestChain, ComputeGraph
