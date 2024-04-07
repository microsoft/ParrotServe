# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

"""Annotations in request."""

from dataclasses import dataclass


@dataclass
class ScheduleAnnotation:
    """Annotations for dispatching Tasks."""

    # This field means this task should not be dispatched to a engine
    # with more than this number of jobs.
    tasks_num_upperbound: int = 256

    # This field means this task should not be dispatched to a engine
    # with more than this number of tokens.
    tokens_num_upperbound: int = 2048

    # Unimplemented
    ddl_requirement: float = 0.0
