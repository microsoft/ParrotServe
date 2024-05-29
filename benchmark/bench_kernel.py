from transformers import AutoTokenizer
import torch
import json

from parrot.engine.builtin.builtin_runner import BuiltinRunner
from parrot.engine.config import BuiltinConfig
from parrot.engine.primitive_job import Fill, Generate
from parrot.sampling_config import SamplingConfig


def bench_decode(
    attn_func: str, batch_size: int, shared_len: int, diverged_len: int, output_len: int
):
    config = BuiltinConfig(
        num_kv_cache_blocks=2000,
        attn_func=attn_func,
        block_size=16,
        max_seq_len=65536,
    )
    sampling_config = SamplingConfig(
        max_gen_length=output_len,
        ignore_tokenizer_eos=True,
    )

    runner = BuiltinRunner("lmsys/vicuna-13b-v1.3", config=config)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    context_len = shared_len + diverged_len

    prompt_token_ids = [[100] * context_len for _ in range(batch_size)]

    shared_fill = Fill(
        session_id=0,
        task_id=0,
        context_id=0,
        parent_context_id=-1,
        token_ids=prompt_token_ids[0][:shared_len],
    )
    diverged_fills = [
        Fill(
            session_id=0,
            task_id=0,
            context_id=i + 1,
            parent_context_id=0,
            token_ids=prompt[shared_len:],
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
    for _ in range(output_len):
        runner.run_iter(gens)

    del runner


if __name__ == "__main__":
    # bench_decode(
    #     attn_func="xformers_fill_vllm_paged_attention_generate",
    #     batch_size=64,
    #     shared_len=8192,
    #     diverged_len=10,
    #     output_len=10,
    # )
    bench_decode(
        attn_func="xformers_fill_shared_prompts_generate",
        batch_size=64,
        shared_len=8192,
        diverged_len=10,
        output_len=10,
    )
