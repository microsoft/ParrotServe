from transformers import AutoTokenizer
import torch
import json

from parrot.engine.builtin.builtin_runner import BuiltinRunner
from parrot.engine.config import BuiltinConfig
from parrot.engine.primitive_job import Fill, Generate
from parrot.sampling_config import SamplingConfig


config = BuiltinConfig(
    num_kv_cache_blocks=2048,
    attn_func="xformers_fill_shared_prompts_generate",
    block_size=16,
    max_seq_len=16384,
)
sampling_config = SamplingConfig(
    max_gen_length=200,
    ignore_tokenizer_eos=True,
)

runner = BuiltinRunner("lmsys/vicuna-7b-v1.3", config=config)
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

with open("../workloads/bingchat/bing_chat_dataset.jsonl", encoding="utf8") as f:
    prompt_token_ids = [
        tokenizer.encode(json.loads(line)["prompt"]) for line in f.readlines()
    ]
prompt_token_ids = prompt_token_ids[:8]
num_seqs = len(prompt_token_ids)


prompts = torch.tensor(prompt_token_ids[0], dtype=torch.int32, device="cuda")
shared_ids = 0
while len(set([prompt[shared_ids] for prompt in prompt_token_ids])) == 1:
    shared_ids += 1

shared_fill = Fill(
    pid=0,
    tid=0,
    context_id=0,
    parent_context_id=-1,
    token_ids=prompt_token_ids[0][:shared_ids],
)
diverged_fills = [
    Fill(
        pid=0,
        tid=0,
        context_id=i + 1,
        parent_context_id=0,
        token_ids=prompt[shared_ids:],
    )
    for i, prompt in enumerate(prompt_token_ids)
]
gens = [
    Generate(
        pid=0,
        tid=0,
        context_id=i + 1,
        parent_context_id=0,
        sampling_config=sampling_config,
    )
    for i, prompt in enumerate(prompt_token_ids)
]

runner.run_iter([shared_fill])
runner.run_iter(diverged_fills)
runner.run_iter(gens[:4])
for _ in range(10):
    runner.run_iter(gens)
