from typing import List, Dict
from transformers import AutoConfig
import torch

from .models.opt import OPTForCausalLM
from .mem import KVContext
from .entity import BackendJob, FillJob, GenerationJob, IterationState
from ..utils import RecyclePool


class Runner:
    """Minimal LLM Runner with adaption to Parrot."""

    def __init__(self, model_name: str):
        # Mgr.
        self.context_mgr: Dict[int, KVContext] = {}
        self.kv_cache_manager = RecyclePool(131072 * 10)  # TODO(chaofan): config this

        # Load Model
        torch.set_default_dtype(torch.float16)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = OPTForCausalLM(self.config)  # Currently only support OPT
        self.model.load_weights(model_name)
        self.model = self.model.cuda()

    def run(self, jobs: List[BackendJob]):
        # Prepare iteration state
        iteration_state = IterationState(
            jobs,
            self.context_mgr,
            self.kv_cache_manager,
            self.config,
            self.model.dtype,
            self.model.device,
        )

        # Convert inputs
        input_ids = []
        input_positions = []

        for job in jobs:
            context = self.context_mgr[job.context_id]
            if isinstance(job, FillJob):
                ids = job.tokens_id
                positions = range(context.context_len, context.context_len + len(ids))
            elif isinstance(job, GenerationJob):
                ids = [context.last_token_id]
                positions = [context.context_len]
            input_ids.extend(ids)
            input_positions.extend(positions)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.int64,
            device=self.model.device,
        )
        input_positions = torch.tensor(
            input_positions,
            dtype=torch.int64,
            device=self.model.device,
        )

        # Execute model
        next_tokens = (
            self.model(input_ids, input_positions, iteration_state).cpu().tolist()
        )
        assert len(next_tokens) == len(jobs)

        # Update context
        for i, token_id in enumerate(next_tokens):
            job = jobs[i]
            context = self.context_mgr[job.context_id]
            context.last_token_id = token_id
