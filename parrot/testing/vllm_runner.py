# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

# Modified from LLMOS-simulator project.


# pylint: disable=missing-module-docstring

import sys
import numpy as np

from typing import List, Optional

from vllm import EngineArgs, LLMEngine, SamplingParams


class vLLMRunner:
    """Runner to execute vLLM (Single GPU)."""

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        max_tokens_sum: int = 81000,
    ):
        # opt_prefix = "facebook/opt-"
        # if model.startswith(opt_prefix):
        #     model_scale = model[len(opt_prefix) :]
        #     hf_config = OPT_CONFIG[model_scale]
        # else:
        #     raise ValueError(f"Currently not support model: {model}")

        self.max_tokens_sum = max_tokens_sum

        if tokenizer is None:
            tokenizer = model

        self.engine_args = EngineArgs(
            model=model,
            tokenizer=model,
            use_dummy_weights=True,
            dtype="float16",
            max_num_seqs=2048,
            max_num_batched_tokens=max_tokens_sum,
        )
        self.llm_engine = LLMEngine.from_engine_args(self.engine_args)

    def reset(self):
        # Reset KV blocks since we only have limited memory
        del self.llm_engine.workers[0].cache_engine
        del self.llm_engine.workers[0].gpu_cache
        self.llm_engine._init_cache()  # pylint: disable=protected-access

    def prefill_random_data(self, batch_size: int, prompt_len: int, output_len: int):
        sampling_params = SamplingParams(max_tokens=output_len, ignore_eos=True)
        for i in range(batch_size):
            self.llm_engine.add_request(
                request_id=str(i),
                prompt=None,
                prompt_token_ids=[
                    np.random.randint(1000, 10000) for _ in range(prompt_len)
                ],
                sampling_params=sampling_params,
            )
        self.llm_engine.workers[0].token_counter = 0
        self.llm_engine.step()

    def step(self):
        self.llm_engine.step()

    def sample_random_data(self, batch_size: int, prompt_len: int, output_len: int):
        self.prefill_random_data(batch_size, prompt_len, output_len)
        self.llm_engine.workers[0].token_counter = 0
        while self.llm_engine.has_unfinished_requests():
            self.llm_engine.step()
