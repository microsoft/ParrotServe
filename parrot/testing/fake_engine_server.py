# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""A fake server for testing.

You can choose whether to connect with OS by the argument: --connect_os.

If you choose to connect with OS, you should start the OS server first: Please 
start the OS server at: http://localhost:9000
"""

import asyncio
import argparse
from dataclasses import asdict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from uvicorn import Config, Server
import time
import numpy as np

from parrot.engine.config import EngineConfig
from parrot.os.engine import EngineRuntimeInfo
from parrot.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_ENGINE_SERVER_PORT,
    ENGINE_HEARTBEAT_INTERVAL,
)
from parrot.protocol.layer_apis import register_engine, engine_heartbeat
from parrot.utils import get_logger, create_task_in_loop

# ---------- Constants ----------
TESTING_RANDOM_SEED = 2333
TESTING_SERVER_HOST = DEFAULT_SERVER_HOST
TESTING_SERVER_PORT = DEFAULT_ENGINE_SERVER_PORT
TESTING_SERVER_URL = f"http://{TESTING_SERVER_HOST}:{TESTING_SERVER_PORT}"
TESTING_FILL_PERTOKEN_TIME = 0.1
TESTING_DECODE_PERTOKEN_TIME = 0.1

OS_URL = "http://localhost:9000"

app = FastAPI()

logger = get_logger("Fake Engine Server")


# Status Data

context = {}  # Context_id -> context_length
num_cached_tokens = 0
num_running_jobs = 0

engine_config = EngineConfig(
    host=TESTING_SERVER_HOST,
    port=TESTING_SERVER_PORT,
    engine_name="Fake Engine",
    tokenizer="facebook/opt-13b",
)


def fake_engine_daemon():
    global num_running_jobs
    global num_cached_tokens

    resp = register_engine(
        http_addr=OS_URL,
        engine_config=engine_config,
    )

    engine_id = resp.engine_id

    while True:
        resp = engine_heartbeat(
            http_addr=OS_URL,
            engine_id=engine_id,
            engine_name=engine_config.engine_name,
            runtime_info=EngineRuntimeInfo(
                num_cached_tokens=num_cached_tokens,
                num_running_jobs=num_running_jobs,
                cache_mem=num_cached_tokens * 4,  # Simple assumption: 4 bytes per token
                model_mem=0,
            ),
        )

        time.sleep(ENGINE_HEARTBEAT_INTERVAL)


@app.post("/fill")
async def fill(request: Request):
    global num_running_jobs
    global num_cached_tokens

    num_running_jobs += 1

    payload = await request.json()

    token_ids = payload["token_ids"]
    text = payload["text"]

    # Suppose the server will always fill all tokens
    # Simulate the time of filling tokens
    if token_ids is not None:
        length = len(token_ids)
    else:
        assert text is not None
        length = len(text.split())

    time.sleep(TESTING_FILL_PERTOKEN_TIME * length)

    num_cached_tokens += length
    context_id = payload["context_id"]
    if context_id not in context:
        context[context_id] = 0
    context[context_id] += length

    num_running_jobs -= 1

    return {
        "filled_len": length,
    }


@app.post("/generate")
async def generate(request: Request):
    global num_running_jobs
    global num_cached_tokens

    num_running_jobs += 1
    payload = await request.json()

    gen_len = int(np.random.exponential(32) + 3)

    time.sleep(TESTING_DECODE_PERTOKEN_TIME * gen_len)

    return {
        "generated_text": "xxx",
        "generated_ids": [],
    }


@app.post("/generate_stream")
async def generate_stream(request: Request):
    global num_running_jobs
    global num_cached_tokens

    num_running_jobs += 1
    payload = await request.json()

    gen_len = int(np.random.exponential(32) + 3)
    # gen_len = 512
    gen_data = np.random.randint(10, 10000, size=(gen_len,)).tolist()

    num_cached_tokens += gen_len
    context_id = payload["context_id"]
    if context_id not in context:
        context[context_id] = 0
    context[context_id] += gen_len

    def generator():
        for data in gen_data:
            # Simulate the time of decoding tokens
            time.sleep(TESTING_DECODE_PERTOKEN_TIME)
            yield data.to_bytes(4, "big")

    num_running_jobs -= 1

    return StreamingResponse(generator())


@app.post("/free_context")
async def free_context(request: Request):
    global num_cached_tokens

    payload = await request.json()
    # assert payload["context_id"] in context

    context_len = 0

    if payload["context_id"] in context:
        num_cached_tokens -= context[payload["context_id"]]
        context_len = context[payload["context_id"]]

    return {
        "context_len": context_len,
    }


@app.post("/ping")
async def ping(request: Request):
    global num_running_jobs
    global num_cached_tokens

    rt_info = EngineRuntimeInfo(
        num_cached_tokens=num_cached_tokens,
        num_running_jobs=num_running_jobs,
        cache_mem=num_cached_tokens * 4,  # Simple assumption: 4 bytes per token
        model_mem=0,
    )

    return {
        "runtime_info": asdict(rt_info),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Engine Server")

    parser.add_argument(
        "--connect_os",
        action="store_true",
        help="Whether to connect with OS.",
    )

    args = parser.parse_args()

    np.random.seed(TESTING_RANDOM_SEED)

    if not args.connect_os:
        uvicorn.run(
            app, host=TESTING_SERVER_HOST, port=TESTING_SERVER_PORT, log_level="info"
        )
    else:
        loop = asyncio.new_event_loop()
        config = Config(
            app=app,
            loop=loop,
            host=TESTING_SERVER_HOST,
            port=TESTING_SERVER_PORT,
            log_level="info",
        )
        uvicorn_server = Server(config)
        create_task_in_loop(fake_engine_daemon(), loop=loop, fail_fast=True)
        loop.run_until_complete(uvicorn_server.serve())
