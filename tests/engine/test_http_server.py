import asyncio
import pytest
import json
import time
from transformers import AutoTokenizer

from parrot.engine.config import EngineConfig
from parrot.protocol.layer_apis import free_context
from parrot.protocol.primitive_request import Fill, Generate
from parrot.os.context.context import Context
from Parrot.parrot.os.engine.engine_node import ExecutionEngine
from parrot.protocol.sampling_config import SamplingConfig
from parrot.testing.get_configs import get_sample_engine_config_path
from parrot.testing.localhost_server_daemon import engine_server


def _test_single_server_simple_serving(config):
    engine_type, config_fn = config
    engine_config_path = get_sample_engine_config_path(config_fn)

    async def main():
        with open(engine_config_path) as f:
            engine_config = dict(json.load(f))
        engine_config.pop("instance")
        engine_config.pop("scheduler")
        engine_config.pop("os")
        engine_config = EngineConfig(**engine_config)

        os_engine = ExecutionEngine(
            engine_id=0,
            config=engine_config,
        )

        ctx = Context(
            context_id=0,
            engine=os_engine,
        )

        prompt_text = "Hello, my name is"
        if engine_config.tokenizer != "unknown":
            tokenizer = AutoTokenizer.from_pretrained(engine_config.tokenizer)
            prompt_tokens = tokenizer(prompt_text)["input_ids"]

            fill_primitive = Fill(
                pid=0,
                tid=0,
                context=ctx,
                token_ids=prompt_tokens,
            )
        else:
            fill_primitive = Fill(
                pid=0,
                tid=0,
                context=ctx,
                text=prompt_text,
            )

        resp = await fill_primitive.apost()
        # assert resp.filled_len == len(prompt_tokens)

        gen_primitive = Generate(
            pid=0,
            tid=0,
            context=ctx,
            sampling_config=SamplingConfig(max_gen_length=10),
        )

        if engine_type == "native":
            generator = gen_primitive.astream()

            text = prompt_text

            async for token_id in generator:
                # print(token_id)
                text += tokenizer.decode([token_id])
        else:
            resp = await gen_primitive.apost()
            text = resp.generated_text

        print("Generated: ", text)

    wait_ready_time = 20 if "vicuna" in config_fn else 5  # seconds

    with engine_server(
        config_fn,
        wait_ready_time=wait_ready_time,
        connect_to_os=False,
    ):
        time.sleep(5)
        asyncio.run(main())


TEST_CONFIGS_LIST = [
    ("native", "opt-125m.json"),
    ("native", "vicuna-7b-v1.3.json"),
    ("openai", "azure-openai-gpt-3.5-turbo.json"),
]


@pytest.mark.skip(reason="OOM in test")
def test_simple_serving():
    for config in TEST_CONFIGS_LIST:
        print("TESTING: ", config)
        _test_single_server_simple_serving(config)
        time.sleep(1)


if __name__ == "__main__":
    test_simple_serving()
