"""A fake server for testing."""

import argparse
import asyncio
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server

from parrot.utils import get_logger, create_task_in_loop
from parrot.protocol.sampling_config import SamplingConfig
from parrot.constants import DEFAULT_SERVER_HOST, DEFAULT_OS_SERVER_PORT

from .engine import NativeExecutionEngine
from ..primitive_job import Fill, Generation


logger = get_logger("Backend Server")

# FastAPI app
app = FastAPI()

# Engine
execution_engine: Optional[NativeExecutionEngine] = None


@app.post("/fill")
async def fill(request: Request):
    payload = await request.json()
    fill_job = Fill(
        pid=payload["pid"],
        tid=payload["tid"],
        context_id=payload["context_id"],
        parent_context_id=payload["parent_context_id"],
        token_ids=payload["token_ids"],
    )
    execution_engine._add_job(fill_job)
    await fill_job.finish_event.wait()

    return {
        "num_filled_tokens": len(fill_job.token_ids),
    }


@app.post("/generate")
async def generate(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    tid = payload["tid"]
    context_id = payload["context_id"]
    parent_context_id = payload["parent_context_id"]
    sampling_config = SamplingConfig(**payload["sampling_config"])

    generation_job = Generation(
        pid=pid,
        tid=tid,
        context_id=context_id,
        parent_context_id=parent_context_id,
        sampling_config=sampling_config,
    )
    execution_engine._add_job(generation_job)

    await generation_job.finish_event.wait()

    generated_token_ids = []
    while not generation_job.output_queue.empty():
        generated_token_ids.append(generation_job.output_queue.get())

    return {
        "generated_text": "",
        "generated_ids": generated_token_ids,
    }


@app.post("/generate_stream")
async def generate_stream(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    tid = payload["tid"]
    context_id = payload["context_id"]
    parent_context_id = payload["parent_context_id"]
    sampling_config = SamplingConfig(**payload["sampling_config"])

    generation_job = Generation(
        pid=pid,
        tid=tid,
        context_id=context_id,
        parent_context_id=parent_context_id,
        sampling_config=sampling_config,
    )
    execution_engine._add_job(generation_job)

    return StreamingResponse(generation_job.generator())


@app.post("/free_context")
async def free_context(request: Request):
    payload = await request.json()
    num_freed_tokens = execution_engine.free_context(
        payload["client_id"], payload["context_id"]
    )

    return {
        "num_freed_tokens": num_freed_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parrot native engine HTTP server")

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file of the native engine.",
        required=True,
    )

    parser.add_argument(
        "--os-address",
        type=str,
        help="HTTP address of the OS.",
        default=f"http://{DEFAULT_SERVER_HOST}:{DEFAULT_OS_SERVER_PORT}",
        required=True,
    )

    args = parser.parse_args()

    # uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    execution_engine = NativeExecutionEngine(args.config_path, args.os_address)

    loop = asyncio.new_event_loop()
    config = Config(
        app=app,
        loop=loop,
        host=execution_engine.engine_config.host,
        port=execution_engine.engine_config.port,
        log_level="info",
    )
    uvicorn_server = Server(config)
    # NOTE(chaofan): We use `fail_fast` because this project is still in development
    # For real deployment, maybe we don't need to quit the backend when there is an error
    create_task_in_loop(execution_engine.engine_loop(), loop=loop, fail_fast=True)
    loop.run_until_complete(uvicorn_server.serve())
