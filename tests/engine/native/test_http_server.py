"""This test requires a running OPT-125M Parrot backend HTTP server.

Use `python3 -m parrot.engine.http_server --config_path configs/engine/native/opt-125m.json --without-os` 
to start a server.

Please use host `localhost` and port `9001` for the server.
Please don't connect to the OS.
"""

import asyncio
import parrot
import json
from transformers import AutoTokenizer
from parrot.engine.config import EngineConfig
from parrot.protocol.layer_apis import free_context
from parrot.protocol.primitive_request import Fill, Generate
from parrot.os.memory.context import Context
from parrot.os.engine import ExecutionEngine
from parrot.protocol.sampling_config import SamplingConfig


def test_simple_serving():
    async def main():
        url = "http://localhost:9001"

        package_path = parrot.__path__[0]
        engine_config_path = package_path + "/../configs/engine/native/opt-125m.json"
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

        prompt_text = "Hello, my name is"
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        prompt_tokens = tokenizer(prompt_text)["input_ids"]

        ctx = Context(
            context_id=0,
            engine=os_engine,
        )

        primitive = Fill(
            pid=0,
            tid=0,
            context=ctx,
            token_ids=prompt_tokens,
        )
        resp = await primitive.apost()
        assert resp.num_filled_len == len(prompt_tokens)

        primitive = Generate(
            pid=0,
            tid=0,
            context=ctx,
            sampling_config=SamplingConfig(max_gen_length=10),
        )
        generator = primitive.astream()

        text = prompt_text

        async for token_id in generator:
            # print(token_id)
            text += tokenizer.decode([token_id])

        resp = free_context(url, 0)
        assert resp.context_len > len(prompt_tokens)

        print("Generated: ", text)

    asyncio.run(main())


if __name__ == "__main__":
    test_simple_serving()
