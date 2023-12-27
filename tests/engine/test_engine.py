import asyncio
import time
from transformers import AutoTokenizer

from parrot.engine.engine_creator import create_engine
from parrot.engine.primitive_job import Fill, Generate
from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import create_task_in_loop
from parrot.testing.get_configs import get_sample_engine_config_path

import torch


def _test_single_engine_simple_serving(config):
    engine_type, config_fn = config

    engine = create_engine(
        engine_config_path=get_sample_engine_config_path(config_fn),
        connect_to_os=False,
    )

    prompt_text = "Hello, my name is"
    tokenizer_name = engine.engine_config.tokenizer
    if tokenizer_name == "unknown":
        fill_job = Fill(
            pid=0,
            tid=0,
            context_id=0,
            parent_context_id=-1,
            text=prompt_text,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        prompt_tokens = tokenizer(prompt_text)["input_ids"]
        fill_job = Fill(
            pid=0,
            tid=0,
            context_id=0,
            parent_context_id=-1,
            token_ids=prompt_tokens,
        )

    gen_job = Generate(
        pid=0,
        tid=0,
        context_id=0,
        parent_context_id=-1,
        sampling_config=SamplingConfig(
            max_gen_length=40,
            ignore_tokenizer_eos=True,
        ),
    )

    async def execute_job(job):
        engine._add_job(job)
        await job.finish_event.wait()

    if engine_type == "builtin":

        async def main():
            create_task_in_loop(engine.engine_loop())
            await execute_job(fill_job)
            await execute_job(gen_job)
            print(tokenizer.decode(gen_job.context.token_ids))

    elif engine_type == "openai":

        async def main():
            create_task_in_loop(engine.engine_loop())
            await execute_job(fill_job)
            await execute_job(gen_job)
            print(gen_job.context.get_latest_context_text())

    try:
        asyncio.run(main(), debug=True)
    except BaseException as e:
        print("Internal error happends:", e)

    del engine
    torch.cuda.empty_cache()


TEST_CONFIGS_LIST = [
    ("builtin", "opt-125m.json"),
    ("builtin", "vicuna-7b-v1.3.json"),
    ("openai", "azure-openai-gpt-3.5-turbo.json"),
]


def test_engine_simple_serving():
    for config in TEST_CONFIGS_LIST:
        print("TESTING: ", config)
        _test_single_engine_simple_serving(config)
        time.sleep(1.0)


if __name__ == "__main__":
    test_engine_simple_serving()
