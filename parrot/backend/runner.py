from typing import List, Dict
from transformers import AutoConfig
import torch

from .models.opt import OPTForCausalLM
from .mem import KVContext
from .entity import BackendJob, FillJob, GenerationJob, IterationState
from ..utils import RecyclePool
from .config import AttentionConfig


class Runner:
    """Minimal LLM Runner with adaption to Parrot."""

    def __init__(self, model_name: str):
        # Mgr.
        self.attn_config = AttentionConfig(
            cache_blocks_num=131072 * 10,  # TODO(chaofan): config this
            attn_func="xformers_with_buffer",
        )
        self.context_manager: Dict[int, KVContext] = {}
        self.kv_cache_manager = RecyclePool(self.attn_config.cache_blocks_num)

        # Load Model
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.model_config = AutoConfig.from_pretrained(model_name)
        torch.set_default_dtype(self.dtype)
        self.model = OPTForCausalLM(
            self.model_config, self.attn_config
        )  # Currently only support OPT
        self.model.load_weights(model_name)
        self.model = self.model.cuda()

    def run(self, jobs: List[BackendJob]):
        # Allocate new context blocks
        for job in jobs:
            # Context
            if job.context_id not in self.context_manager:
                assert isinstance(job, FillJob)
                if job.parent_context_id not in self.context_manager:
                    assert job.parent_context_id == -1
                    parent_context = None
                else:
                    parent_context = self.context_manager[job.parent_context_id]
                self.context_manager[job.context_id] = KVContext(
                    job.context_id, parent_context
                )

            context = self.context_manager[job.context_id]

            # Allocate blocks
            allocated_blocks_id: List[int] = []

            if isinstance(job, FillJob):
                for _ in range(len(job.tokens_id)):
                    allocated_blocks_id.append(self.kv_cache_manager.allocate())
            elif isinstance(job, GenerationJob):
                allocated_blocks_id.append(self.kv_cache_manager.allocate())

            context.tokens_kv_block_id.extend(allocated_blocks_id)

        # Prepare iteration state
        iteration_state = IterationState(
            jobs,
            self.context_manager,
            self.model_config,
            self.attn_config,
            self.dtype,
            self.device,
        )

        # Convert inputs
        input_ids = []
        input_positions = []

        for job in jobs:
            context = self.context_manager[job.context_id]
            context_len = context.get_context_len()
            if isinstance(job, FillJob):
                input_ids.extend(job.tokens_id)
                input_positions.extend(
                    range(context_len - len(job.tokens_id), context_len)
                )
            elif isinstance(job, GenerationJob):
                input_ids.append(context.tokens_id[-1])
                input_positions.append(context_len - 1)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.int64,
            device=self.device,
        )
        input_positions = torch.tensor(
            input_positions,
            dtype=torch.int64,
            device=self.device,
        )

        # Execute model
        next_tokens = (
            self.model(input_ids, input_positions, iteration_state).cpu().tolist()
        )
        assert len(next_tokens) == len(jobs)

        # Update context
        for i, token_id in enumerate(next_tokens):
            job = jobs[i]
            context = self.context_manager[job.context_id]
            context.tokens_id.append(token_id)
