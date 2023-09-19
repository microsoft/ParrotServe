from typing import List, Dict
from transformers import AutoConfig
import torch

from .model_instantiation import instantiate_model
from .mem import KVContext, init_model_cache_storage
from .iter_state import BackendPrimitiveJob, Fill, Generation, IterationState
from ..utils import RecyclePool, set_random_seed
from .config import RunnerConfig


class Runner:
    """Minimal LLM Runner with adaption to Parrot."""

    def __init__(self, config: RunnerConfig):
        # self.config = RunnerConfig(
        #     num_kv_cache_blocks=500,  # TODO(chaofan): config this
        #     attn_func="xformers_with_buffer",
        #     random_seed=0,
        # )

        self.runner_config = config
        self.context_manager: Dict[int, KVContext] = {}
        self.kv_cache_manager = RecyclePool(self.runner_config.num_kv_cache_blocks)

        # Load Model
        self.hf_model_config = AutoConfig.from_pretrained(self.runner_config.model_name)
        self.model = instantiate_model(self.hf_model_config, self.runner_config)

        # Init model cache storage
        init_model_cache_storage(self.hf_model_config, self.runner_config)

        # Set random seed
        set_random_seed(self.runner_config.random_seed)

    def bind_job_context(self, job: BackendPrimitiveJob):
        if job.context_id not in self.context_manager:
            assert isinstance(job, Fill)
            if job.parent_context_id not in self.context_manager:
                assert job.parent_context_id == -1
                parent_context = None
            else:
                parent_context = self.context_manager[job.parent_context_id]
            self.context_manager[job.context_id] = KVContext(
                job.context_id, parent_context, self.kv_cache_manager
            )
        job.context = self.context_manager[job.context_id]

    @torch.inference_mode()
    def run_iter(self, jobs: List[BackendPrimitiveJob]):
        # Allocate new context blocks
        for job in jobs:
            # NOTE(chaofan): if we use engine, this is not necessary.
            if job.context is None:
                self.bind_job_context(job)

            # Allocate blocks
            allocated_blocks_id: List[int] = []

            if isinstance(job, Fill):
                job.context.token_ids.extend(job.token_ids)
                job.context.allocate(len(job.token_ids))
                job.context.last_extended_by_fill = True
            elif isinstance(job, Generation):
                job.context.allocate(1)
                if job.context.last_extended_by_fill:
                    # NOTE: See parrot/backend/mem.py: L31.
                    job.put_token(job.context.get_last_token_id())
                job.context.last_extended_by_fill = False

            job.context.tokens_kv_block_id.extend(allocated_blocks_id)

        # Prepare iteration state
        iteration_state = IterationState(
            jobs,
            self.hf_model_config,
            self.runner_config,
        )

        # Convert inputs
        input_ids = []
        input_positions = []

        for job in jobs:
            context = self.context_manager[job.context_id]
            context_len = context.get_context_len()
            if isinstance(job, Fill):
                input_ids.extend(job.token_ids)
                input_positions.extend(
                    range(context_len - len(job.token_ids), context_len)
                )
            elif isinstance(job, Generation):
                input_ids.append(context.get_last_token_id())
                input_positions.append(context_len - 1)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.int64,
            device=self.runner_config.device,
        )
        input_positions = torch.tensor(
            input_positions,
            dtype=torch.int64,
            device=self.runner_config.device,
        )

        # Execute model
        next_tokens = (
            self.model(input_ids, input_positions, iteration_state).cpu().tolist()
        )
        assert len(next_tokens) == len(jobs)

        # Update context
        for i, token_id in enumerate(next_tokens):
            job = jobs[i]
            assert job.context is not None, "Context should be assigned."
            job.context.token_ids.append(token_id)

            # Mark finish flag
            if isinstance(job, Fill):
                job.finish_event.set()
            elif isinstance(job, Generation):
                job.put_token(token_id)
                if job.check_stop():
                    job.finish_event.set()
