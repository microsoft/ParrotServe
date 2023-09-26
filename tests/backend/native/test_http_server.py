"""This test requires a running Parrot HTTP server.

Use `python3 -m parrot.backend.native.http_server --config_path [engine_config_path]` to start a server.
Please use host `localhost` and port `8888` for the server.
"""

import asyncio
from transformers import AutoTokenizer
from parrot.protocol import *


def test_simple_serving():
    async def main():
        prompt_text = "Hello, my name is"
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        prompt_tokens = tokenizer(prompt_text)["input_ids"]

        resp = check_heartbeat("http://localhost:8888", "test", "0")
        assert resp.num_cached_tokens == 0
        assert resp.cached_tokens_size == 0
        assert resp.num_running_jobs == 0

        resp = fill("http://localhost:8888", "0", 0, -1, prompt_tokens)
        assert resp.num_filled_tokens == len(prompt_tokens)

        generator = agenerate(
            "http://localhost:8888",
            "0",
            0,
            0,
            -1,
            SamplingParams(max_gen_length=10),
        )

        text = prompt_text

        async for token_id in generator:
            # print(token_id)
            text += tokenizer.decode([token_id])

        resp = free_context("http://localhost:8888", "0", 0)
        assert resp.num_freed_tokens > len(prompt_tokens)

        print("Generated: ", text)

    asyncio.run(main())


if __name__ == "__main__":
    test_simple_serving()
