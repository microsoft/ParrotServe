from typing import List
import torch
from torch import nn
from transformers import PretrainedConfig

from parrot.protocol.sampling_config import SamplingConfig


class Sampler(nn.Module):
    def __init__(self, config: PretrainedConfig, embd_weight: torch.Tensor):
        super().__init__()
        self.embd_weight = embd_weight  # It's a reference
        self.vocab_size = config.vocab_size

    def forward(
        self, hidden_states: torch.Tensor, sampling_config: List[SamplingConfig]
    ):
        if hidden_states.shape[0] == 0:
            return torch.zeros(0, dtype=torch.int64, device=hidden_states.device)

        assert hidden_states.shape[0] == len(sampling_config)

        logits = torch.matmul(hidden_states, self.embd_weight.t())
        # ids = torch.ones(probs.shape[0], dtype=torch.int64, device=probs.device)

        # Applying temperature scaling
        temperature = [sp.temperature for sp in sampling_config]
        temperature = torch.tensor(
            temperature, dtype=torch.float, device=logits.device
        ).unsqueeze(-1)
        logits.div_(temperature)

        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)

        return ids
