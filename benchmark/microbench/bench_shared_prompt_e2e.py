import json
import time

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from parrot.engine.builtin.builtin_runner import BuiltinRunner
from parrot.engine.config import BuiltinConfig
from parrot.engine.primitive_job import Fill, Generate
from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import torch_profile, cprofile
from parrot.engine.builtin.mem import get_k_cache, get_v_cache


TOKEN_NUMS = [
    (6014, 619), (5986, 393), (5508, 183), (5573, 191),
    (5986, 393), (5708, 209), (5709, 212), (5636, 192),
    (6943, 800), (5961, 360), (5593, 192), (5757, 232),
    (5757, 232), (5573, 191), (5885, 351), (5885, 351),
    (6014, 619), (5573, 191), (5986, 393), (5765, 248),
    (5765, 248), (5961, 360), (5961, 360), (5986, 393),
    (5708, 209), (5757, 232), (5749, 232), (6943, 800),
    (5961, 360), (5809, 269), (5961, 360), (5653, 195),
    (6066, 800), (5986, 393), (5809, 269), (5800, 256),
    (5757, 232), (5846, 303), (5809, 269), (5708, 209),
    (5783, 251), (5708, 209), (6943, 800), (5508, 183),
    (6066, 800), (5593, 192), (5986, 393), (5593, 192),
    (5709, 212), (5856, 313), (6943, 800), (5667, 196),
    (5653, 195), (5709, 212), (5653, 195), (5885, 351),
    (5986, 393), (5885, 351), (5757, 232), (5783, 251),
    (5749, 232), (5667, 196), (5885, 351), (5961, 360),
]


if __name__ == '__main__':
    np.random.seed(2023)
    torch.manual_seed(2023)

    batch_size = 64
    config = BuiltinConfig(
        num_kv_cache_blocks=6800,
        # attn_func="xformers_fill_vllm_paged_attention_generate",
        attn_func="xformers_fill_shared_prompts_generate",
        block_size=16,
        max_seq_len=8192,
    )
    sampling_config = SamplingConfig(
        max_gen_length=800,
        ignore_tokenizer_eos=True,
    )

    # model_path = '/models/llama-hf/llama-7b_hf'
    model_path = '/models/vicuna/vicuna-7b-v1.3'
    runner = BuiltinRunner(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open("./bing_chat_dataset.jsonl", encoding="utf8") as f:
        prompt_token_ids = [tokenizer.encode(json.loads(line)['prompt']) for line in f.readlines()]

    shared_ids = 0
    while len(set([prompt[shared_ids] for prompt in prompt_token_ids])) == 1:
        shared_ids += 1
    print(f'Shared token num: {shared_ids}')

    np.random.shuffle(prompt_token_ids)
    prompt_token_ids = prompt_token_ids[:len(TOKEN_NUMS)]

    shared_fill = Fill(pid=0, tid=0, context_id=0, parent_context_id=-1, token_ids=prompt_token_ids[0][:shared_ids])
    e2e_time_sf, model_time_sf = runner.run_iter([shared_fill])

    start_time = time.perf_counter_ns()

    e2e_time_df, model_time_df, e2e_time_gen, model_time_gen = 0, 0, 0, 0
    for stage, start_seq_idx in enumerate(range(0, len(prompt_token_ids), batch_size)):
        end_seq_idx = start_seq_idx + batch_size
        diverged_fills = [
            Fill(pid=0, tid=0, context_id=i + 1, parent_context_id=0, token_ids=prompt[shared_ids:])
            for i, prompt in enumerate(prompt_token_ids[start_seq_idx:end_seq_idx])
        ]
        gens = [
            Generate(pid=0, tid=0, context_id=i + 1, parent_context_id=0, sampling_config=sampling_config)
            for i, prompt in enumerate(prompt_token_ids[start_seq_idx:end_seq_idx])
        ]
        et, mt = runner.run_iter(diverged_fills)
        e2e_time_df += et
        model_time_df += mt
        num = 0
        while len(gens) > 0:
            et, mt = runner.run_iter(gens)
            e2e_time_gen += et
            model_time_gen += mt
            # gens = [gen for gen in gens if not gen.check_stop()]
            gens = [gen for gen in gens if num < TOKEN_NUMS[gen.context_id - 1][1]]
            num += 1
            print(f'#{stage}: {num:>2}')
        for context_id in range(1, batch_size + 1):
            runner.context_manager.free_context(context_id)

    end_time = time.perf_counter_ns()
    total_time = end_time - start_time
    print(f'  Shared Fill Time: {e2e_time_sf / 1e9:7.3f} s, {model_time_sf / 1e9:7.3f} s')
    print(f'Diverged Fill Time: {e2e_time_df / 1e9:7.3f} s, {model_time_df / 1e9:7.3f} s')
    print(f'   Generation Time: {e2e_time_gen / 1e9:7.3f} s, {model_time_gen / 1e9:7.3f} s')
    print(f'        Total Time: {total_time / 1e9:7.3f} s')
