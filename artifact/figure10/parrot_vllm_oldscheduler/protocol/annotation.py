# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

"""Annotations for Semantic Variables."""

from dataclasses import dataclass


@dataclass
class DispatchAnnotation:
    """Annotations for dispatching LLM calls."""

    # This field means this request should not be dispatched to a engine with more than this number of jobs.
    requests_num_upperbound: int = 256
    tokens_num_upperbound: int = 2048
    ddl_requirement: float = 0.0
