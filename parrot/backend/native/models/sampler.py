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
        temperature = [sf.temperature for sf in sampling_config]
        temperature = torch.tensor(
            temperature, dtype=torch.float, device=logits.device
        ).unsqueeze(-1)
        logits.div_(temperature)

        sorted_logits, logits_idx = logits.sort(dim=-1, descending=True)

        # Applying top_p
        top_ps = [sf.top_p for sf in sampling_config]
        top_ps = torch.tensor(
            top_ps, dtype=torch.float, device=logits.device
        ).unsqueeze(-1)
        sorted_probs = sorted_logits.softmax(dim=-1)
        sum_probs = sorted_probs.cumsum(dim=-1)
        mask = (sum_probs - sorted_probs) > top_ps
        sorted_logits[mask] = -float("inf")

        logits = torch.gather(
            sorted_logits, dim=-1, index=torch.argsort(logits_idx, dim=-1)
        )

        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)

        return ids
