from transformers import AutoTokenizer
import torch
import json

from parrot.engine.builtin.builtin_runner import BuiltinRunner
from parrot.engine.config import BuiltinConfig
from parrot.engine.primitive_job import Fill, Generate
from parrot.sampling_config import SamplingConfig

def test_shared_decode():
    config = BuiltinConfig(
        num_kv_cache_blocks=2048,
        # attn_func="xformers_fill_shared_prompts_generate",
        attn_func="xformers_fill_vllm_paged_attention_generate",
        block_size=16,
        max_seq_len=16384,
    )
    sampling_config = SamplingConfig(
        max_gen_length=200,
        ignore_tokenizer_eos=True,
    )

    runner = BuiltinRunner("lmsys/vicuna-7b-v1.3", config=config)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    # bs=2
    # shared len = 1712 
    # diverged len = 3
    prompt_token_ids = [
        [100] * 1712 + [200, 300, 400],
        [100] * 1712 + [300, 400, 500],
    ]
    num_seqs = len(prompt_token_ids)

    shared_ids = 0
    while len(set([prompt[shared_ids] for prompt in prompt_token_ids])) == 1:
        shared_ids += 1
    print(shared_ids)

    shared_fill = Fill(
        session_id=0,
        task_id=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=prompt_token_ids[0][:shared_ids],
    )
    diverged_fills = [
        Fill(
            session_id=0,
            task_id=0,
            context_id=i + 1,
            parent_context_id=0,
            token_ids=prompt[shared_ids:],
        )
        for i, prompt in enumerate(prompt_token_ids)
    ]
    gens = [
        Generate(
            session_id=0,
            task_id=0,
            context_id=i + 1,
            parent_context_id=0,
            sampling_config=sampling_config,
        )
        for i, prompt in enumerate(prompt_token_ids)
    ]

    runner.run_iter([shared_fill])
    runner.run_iter(diverged_fills)
    for _ in range(10):
        runner.run_iter(gens)


if __name__ == "__main__":
    test_shared_decode()