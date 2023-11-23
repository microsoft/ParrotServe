# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

"""Annotations for Semantic Variables."""

from dataclasses import dataclass


@dataclass
class DispatchAnnotation:
    """Annotations for dispatching LLM calls."""

    max_jobs_num: int = 256
