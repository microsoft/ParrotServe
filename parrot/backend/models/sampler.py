from typing import List
import torch
from torch import nn
from transformers import PretrainedConfig

from ..iter_state import IterationState


class Sampler(nn.Module):
    def __init__(self, config: PretrainedConfig, embd_weight: torch.Tensor):
        super().__init__()
        self.embd_weight = embd_weight  # It's a reference
        self.vocab_size = config.vocab_size

    def forward(self, hidden_states: torch.Tensor, iteration_state: IterationState):
        # Get last tokens
        idx = 0
        indicies: List[int] = []
        for n in iteration_state.num_fill_tokens:
            idx += n
            indicies.append(idx - 1)
        for _ in range(iteration_state.num_generation_jobs):
            idx += 1
            indicies.append(idx - 1)

        hidden_states = hidden_states[indicies]

        logits = torch.matmul(hidden_states, self.embd_weight.t())
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)

        # TODO(chaofan): Apply top-k sampling, temperature, etc.

        return ids
