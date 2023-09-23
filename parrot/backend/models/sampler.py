from typing import List
import torch
from torch import nn
from transformers import PretrainedConfig

from ...protocol.sampling_params import SamplingParams


class Sampler(nn.Module):
    def __init__(self, config: PretrainedConfig, embd_weight: torch.Tensor):
        super().__init__()
        self.embd_weight = embd_weight  # It's a reference
        self.vocab_size = config.vocab_size

    def forward(
        self, hidden_states: torch.Tensor, sampling_params: List[SamplingParams]
    ):
        if hidden_states.shape[0] == 0:
            return torch.zeros(0, dtype=torch.int64, device=hidden_states.device)

        assert hidden_states.shape[0] == len(sampling_params)

        logits = torch.matmul(hidden_states, self.embd_weight.t())
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
        # ids = torch.ones(probs.shape[0], dtype=torch.int64, device=probs.device)

        # TODO(chaofan): Apply top-k sampling, temperature, etc.

        return ids
