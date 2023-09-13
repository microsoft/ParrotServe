from typing import List
import torch
from torch import nn
from transformers import OPTConfig

from ..entity import IterationState


class GreedySampler(nn.Module):
    def __init__(self, config: OPTConfig, embd_weight: torch.Tensor):
        super().__init__()
        self.embd_weight = embd_weight
        self.vocab_size = config.vocab_size

    def forward(self, hidden_states: torch.Tensor, metadata: IterationState):
        # Get last tokens
        idx = 0
        indicies: List[int] = []
        for i in range(len(metadata.seq_ids)):
            idx += metadata.prefill_lens[i]
            indicies.append(idx - 1)
        hidden_states = hidden_states[indicies]

        logits = torch.matmul(hidden_states, self.embd_weight.t())
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        logprobs = torch.log(probs)
        ids = torch.argmax(logprobs, dim=-1)
        return ids
