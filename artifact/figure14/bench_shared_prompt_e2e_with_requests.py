import json
import time
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from parrot.engine.builtin.builtin_runner import BuiltinRunner
from parrot.engine.config import BuiltinConfig
from parrot.engine.primitive_job import PrimitiveJob, Fill, Generate
from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import torch_profile, cprofile
from parrot.engine.builtin.mem import get_k_cache, get_v_cache


parser = argparse.ArgumentParser(description="shared prompt end-to-end benchmark")
parser.add_argument(
    "-m",
    "--mode",
    default="parrot_shared",
    choices=["vllm_diverged", "vllm_shared", "parrot_shared"],
    help="attention mode",
)
parser.add_argument(
    "-b",
    "--batch",
    default=0,
    type=int,
    help="batch size",
)
parser.add_argument(
    "-l",
    "--len",
    default=800,
    type=int,
    help="max generated tokens",
)
parser.add_argument(
    "-s",
    "--use-sample",
    action="store_true",
    help="use sampled token num list",
)
parser.add_argument(
    "--log-path",
    help="log file path",
)
args = parser.parse_args()


MODEL_NAME = "lmsys/vicuna-7b-v1.3"
DATA_PATH = "../workloads/bingchat/bing_chat_dataset_prompt_len.txt"
MAX_BLOCK_NUM = 6800
SAMPLED_TOKEN_NUMS = [
    (6014, 619),
    (5986, 393),
    (5508, 183),
    (5573, 191),
    (5986, 393),
    (5708, 209),
    (5709, 212),
    (5636, 192),
    (6943, 800),
    (5961, 360),
    (5593, 192),
    (5757, 232),
    (5757, 232),
    (5573, 191),
    (5885, 351),
    (5885, 351),
    (6014, 619),
    (5573, 191),
    (5986, 393),
    (5765, 248),
    (5765, 248),
    (5961, 360),
    (5961, 360),
    (5986, 393),
    (5708, 209),
    (5757, 232),
    (5749, 232),
    (6943, 800),
    (5961, 360),
    (5809, 269),
    (5961, 360),
    (5653, 195),
    (6066, 800),
    (5986, 393),
    (5809, 269),
    (5800, 256),
    (5757, 232),
    (5846, 303),
    (5809, 269),
    (5708, 209),
    (5783, 251),
    (5708, 209),
    (6943, 800),
    (5508, 183),
    (6066, 800),
    (5593, 192),
    (5986, 393),
    (5593, 192),
    (5709, 212),
    (5856, 313),
    (6943, 800),
    (5667, 196),
    (5653, 195),
    (5709, 212),
    (5653, 195),
    (5885, 351),
    (5986, 393),
    (5885, 351),
    (5757, 232),
    (5783, 251),
    (5749, 232),
    (5667, 196),
    (5885, 351),
    (5961, 360),
]
RANDOM_SEED = 2023


class FIFOContextPool(object):

    def __init__(
        self, runner: BuiltinRunner, sampling_config: SamplingConfig, size: int
    ):
        self._jobs: list[PrimitiveJob | None] = [None] * size
        self._gen_limits: list[int] = [65536] * size
        self._runner = runner
        self._sampling_config = sampling_config
        self.e2e_time = 0
        self.model_time = 0
        self._start_time = [0] * size
        self.request_latency_list = []

    def push(self, prompt_token_ids: list[int], parent_context_id: int, gen_limit: int):
        context_idx = self._jobs.index(None)
        self._jobs[context_idx] = Fill(
            pid=0,
            tid=0,
            context_id=context_idx,
            parent_context_id=parent_context_id,
            token_ids=prompt_token_ids,
        )
        self._gen_limits[context_idx] = gen_limit
        self._start_time[context_idx] = time.perf_counter_ns()

    def run(self):
        num_voids = 0
        et, mt = self._runner.run_iter([job for job in self._jobs if job is not None])
        self.e2e_time += et
        self.model_time += mt
        for context_idx, job in enumerate(self._jobs):
            if job is None:
                num_voids += 1
            else:
                if isinstance(job, Fill):
                    self._jobs[context_idx] = Generate(
                        pid=0,
                        tid=0,
                        context_id=job.context_id,
                        parent_context_id=job.parent_context_id,
                        sampling_config=self._sampling_config,
                    )
                else:
                    self._gen_limits[context_idx] -= 1
                    if self._gen_limits[context_idx] <= 0:
                        self.request_latency_list.append(
                            (time.perf_counter_ns() - self._start_time[context_idx])
                            / 1e9
                        )
                        self._runner.context_manager.free_context(context_idx)
                        self._jobs[context_idx] = None
                        num_voids += 1
        return num_voids


def profile_bing_chat(
    shared: bool,
    use_sample: bool,
    max_gen_length: int,
    batch_size: int,
    attn_func: str,
):
    config = BuiltinConfig(
        num_kv_cache_blocks=MAX_BLOCK_NUM,
        attn_func=attn_func,
        block_size=16,
        max_seq_len=8192,
    )
    sampling_config = SamplingConfig(
        max_gen_length=max_gen_length,
        ignore_tokenizer_eos=True,
    )

    runner = BuiltinRunner(MODEL_NAME, config=config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(DATA_PATH, encoding="utf8") as f:
        # prompt_token_ids = [
        #     tokenizer.encode(json.loads(line)["prompt"]) for line in f.readlines()
        # ]
        lines = f.read().splitlines()
        prompt_token_ids = [[666] * int(line) for line in lines]  # dummy token id

    shared_ids = 0
    if shared:
        parent_context_id = batch_size
        # while len(set([prompt[shared_ids] for prompt in prompt_token_ids])) == 1:
        #     shared_ids += 1
        shared_ids = 5107
    else:
        parent_context_id = -1
    print(f"Shared token num: {shared_ids}")

    if use_sample:
        np.random.shuffle(prompt_token_ids)
        prompt_token_ids = prompt_token_ids[: len(SAMPLED_TOKEN_NUMS)]
        gen_limits = [num[1] for num in SAMPLED_TOKEN_NUMS]
    else:
        gen_limits = [max_gen_length] * len(prompt_token_ids)

    context_pool = FIFOContextPool(runner, sampling_config, batch_size)

    start_time = time.perf_counter_ns()

    if shared:
        shared_fill = Fill(
            pid=0,
            tid=0,
            context_id=batch_size,
            parent_context_id=-1,
            token_ids=prompt_token_ids[0][:shared_ids],
        )
        e2e_time_sf, model_time_sf = runner.run_iter([shared_fill])
    else:
        e2e_time_sf, model_time_sf = 0, 0

    for prompt_idx in range(batch_size):
        context_pool.push(
            prompt_token_ids[prompt_idx][shared_ids:],
            parent_context_id,
            gen_limits[prompt_idx],
        )
    num_voids = 0
    while len(context_pool.request_latency_list) < len(prompt_token_ids):
        num_voids = context_pool.run()
        print(
            f"[#{prompt_idx + num_voids + 1 - batch_size:0>2} - #{prompt_idx:0>2}] / {len(prompt_token_ids)}"
        )
        for _ in range(min(num_voids, len(prompt_token_ids) - prompt_idx)):
            context_pool.push(
                prompt_token_ids[prompt_idx][shared_ids:],
                parent_context_id,
                gen_limits[prompt_idx],
            )
            prompt_idx += 1

    e2e_time_dfg = context_pool.e2e_time
    model_time_dfg = context_pool.model_time
    end_time = time.perf_counter_ns()
    total_time = end_time - start_time

    print(
        f"  Shared Fill      : {e2e_time_sf / 1e9:7.3f} s, {model_time_sf / 1e9:7.3f} s"
    )
    print(
        f"Diverged Fill + Gen: {e2e_time_dfg / 1e9:7.3f} s, {model_time_dfg / 1e9:7.3f} s"
    )
    print(f"              Total: {total_time / 1e9:7.3f} s")
    return (
        e2e_time_sf,
        model_time_sf,
        e2e_time_dfg,
        model_time_dfg,
        total_time,
    ), context_pool.request_latency_list


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    shared = args.mode.endswith("shared")
    use_sample = args.use_sample
    max_gen_length = args.len
    batch_size = args.batch
    log_path = args.log_path
    if batch_size <= 0:
        batch_size = 64
    if args.mode.startswith("vllm"):
        attn_func = "xformers_fill_vllm_paged_attention_generate"
    elif args.mode.startswith("parrot"):
        attn_func = "xformers_fill_shared_prompts_generate"

    if use_sample:
        params = [args.mode, batch_size]
    else:
        params = [args.mode, max_gen_length]
    params = ", ".join([str(p) for p in params])

    try:
        results, latency_list = profile_bing_chat(
            shared, use_sample, max_gen_length, batch_size, attn_func
        )
    except ValueError as e:
        results, latency_list = [np.nan] * 5, [np.nan]
        print(e)
    results = ", ".join([str(r / 1e9) for r in results])
    latency = "+".join([str(x) for x in latency_list])

    with open(log_path, "a") as f:
        f.write(f"{params}, {results}, {latency}\n")
