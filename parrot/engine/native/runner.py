from typing import List, Dict
from transformers import AutoConfig
import torch
import time

from parrot.utils import RecyclePool, set_random_seed, get_logger
from parrot.constants import NONE_CONTEXT_ID
from parrot.protocol.sampling_config import SamplingConfig

from .model_instantiation import instantiate_model
from .mem import init_model_cache_storage
from .block_context import BlockContext
from .iter_state import IterationState
from ..low_level_context import ContextManager
from ...protocol.primitives.job import PrimitiveJob, Fill, Generation
from ..config import NativeConfig


logger = get_logger("Runner")


class Runner:
    """Minimal LLM Runner with adaption to Parrot."""

    def __init__(self, config: NativeConfig):
        self.native_config = config
        self.context_manager = ContextManager()
        self.kv_cache_manager = RecyclePool(self.native_config.num_kv_cache_blocks)

        # Load Model
        self.hf_model_config = AutoConfig.from_pretrained(self.native_config.model_name)
        self.model = instantiate_model(self.hf_model_config, self.native_config)

        # Init model cache storage
        init_model_cache_storage(self.hf_model_config, self.native_config)

        # Set random seed
        set_random_seed(self.native_config.random_seed)

    @torch.inference_mode()
    def run_iter(self, jobs: List[PrimitiveJob]) -> (int, int):
        logger.debug(f"Running {len(jobs)} jobs. ")

        st = time.perf_counter_ns()

        # We should sort jobs such that Fill jobs are before Generation jobs.
        jobs.sort(key=lambda job: isinstance(job, Generation))

        # Some generation jobs should do "first sampling"
        first_sampling_states: List[torch.Tensor] = []
        first_sampling_config: List[SamplingConfig] = []
        first_sampling_jobs: List[Generation] = []

        # Allocate new context blocks
        for job in jobs:
            # NOTE(chaofan): if we use engine, this is not necessary.
            if job.context is None:
                self.context_manager.bind_job_context(
                    job, BlockContext, kv_cache_manager=self.kv_cache_manager
                )

            # Allocate blocks
            allocated_blocks_id: List[int] = []

            if isinstance(job, Fill):
                job.context.token_ids.extend(job.token_ids)
                job.context.allocate(len(job.token_ids))
            elif isinstance(job, Generation):
                job.context.allocate(1)
                last_hidden_state = job.context.get_last_hidden_state()
                if last_hidden_state is not None:
                    first_sampling_states.append(last_hidden_state)
                    first_sampling_config.append(job.sampling_config)
                    first_sampling_jobs.append(job)
                    job.context.last_hidden_state = None

            job.context.token_kv_block_ids.extend(allocated_blocks_id)

        # First sampling
        if len(first_sampling_states) > 0:
            first_sampling_states = torch.stack(first_sampling_states)
            first_sampling_tokens = (
                self.model.sampler(first_sampling_states, first_sampling_config)
                .cpu()
                .tolist()
            )
            for i, job in enumerate(first_sampling_jobs):
                job.put_token(first_sampling_tokens[i])

        # Prepare iteration state
        iteration_state = IterationState(
            jobs,
            self.hf_model_config,
            self.native_config,
        )

        # Convert inputs
        input_ids = []
        input_positions = []

        for job in jobs:
            context_len = job.context.get_context_len()
            if isinstance(job, Fill):
                input_ids.extend(job.token_ids)
                input_positions.extend(
                    range(context_len - len(job.token_ids), context_len)
                )
            elif isinstance(job, Generation):
                input_ids.append(job.context.get_last_token_id())
                input_positions.append(context_len - 1)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.int64,
            device=self.native_config.device,
        )
        input_positions = torch.tensor(
            input_positions,
            dtype=torch.int64,
            device=self.native_config.device,
        )

        st_model = time.perf_counter_ns()
        # Execute model
        fill_hidden_states, next_tokens = self.model(
            input_ids, input_positions, iteration_state
        )
        next_tokens = next_tokens.cpu().tolist()
        ed_model = time.perf_counter_ns()
        assert fill_hidden_states.shape[0] + len(next_tokens) == len(jobs)

        # Update context
        for i, job in enumerate(jobs):
            assert job.context is not None, "Context should be assigned."
            if isinstance(job, Fill):
                job.context.last_hidden_state = fill_hidden_states[i]
                job.finish_event.set()
            elif isinstance(job, Generation):
                token_id = next_tokens[i - iteration_state.num_fill_jobs]
                job.put_token(token_id)
                if job.check_stop():
                    job.finish_event.set()

        ed = time.perf_counter_ns()

        e2e_time = (ed - st) / 1e9
        model_time = (ed_model - st_model) / 1e9
        logger.debug(
            f"Finished running {len(jobs)} jobs. "
            f"({iteration_state.num_fill_jobs} Fills, {iteration_state.num_generation_jobs} Generations). "
            f"Total Time used: {e2e_time} (s); "
            f"Model Time used: {model_time} (s)."
        )

        return e2e_time, model_time
