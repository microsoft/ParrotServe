# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List
import torch
from ..iter_state import IterationState


def hidden_states_postprocess(
    hidden_states: torch.Tensor, iteration_state: IterationState
):
    """Postprocess hidden states."""

    idx = 0
    indicies: List[int] = []
    for n in iteration_state.num_fill_tokens:
        idx += n
        indicies.append(idx - 1)

    return hidden_states[indicies], hidden_states[idx:]
