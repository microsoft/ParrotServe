# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import asyncio
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server

from parrot.utils import get_logger, create_task_in_loop

from .engine_creator import create_engine
from .llm_engine import LLMEngine

logger = get_logger("Backend Server")

# FastAPI app
app = FastAPI()

# Engine
llm_engine: Optional[LLMEngine] = None


@app.post("/fill")
async def fill(request: Request):
    payload = await request.json()
    logger.debug(f"Received fill request from pid={payload['pid']}")
    return await llm_engine.fill(payload)


@app.post("/generate")
async def generate(request: Request):
    payload = await request.json()
    logger.debug(f"Received generate request from pid={payload['pid']}")
    return await llm_engine.generate(payload)


@app.post("/generate_stream")
async def generate_stream(request: Request):
    payload = await request.json()
    logger.debug(f"Received generate_stream request from pid={payload['pid']}")
    return StreamingResponse(llm_engine.generate_stream(payload))


@app.post("/free_context")
async def free_context(request: Request):
    payload = await request.json()
    logger.debug(f"Received free_context request")
    return llm_engine.free_context(payload)


@app.post("/ping")
async def ping(request: Request):
    return {}


def start_server(engine_config_path: str, connect_to_os: bool = True):
    global llm_engine
    global app

    llm_engine = create_engine(
        engine_config_path=engine_config_path,
        connect_to_os=connect_to_os,
    )

    loop = asyncio.new_event_loop()
    config = Config(
        app=app,
        loop=loop,
        host=llm_engine.engine_config.host,
        port=llm_engine.engine_config.port,
        log_level="info",
    )
    uvicorn_server = Server(config)
    # NOTE(chaofan): We use `fail_fast` because this project is still in development
    # For real deployment, maybe we don't need to quit the backend when there is an error
    create_task_in_loop(llm_engine.engine_loop(), loop=loop, fail_fast=True)
    loop.run_until_complete(uvicorn_server.serve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parrot native engine HTTP server")

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file of the native engine.",
        required=True,
    )

    parser.add_argument(
        "--without-os",
        action="store_true",
        help="Whether to start the engine without connecting to OS.",
    )

    args = parser.parse_args()

    # uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    start_server(args.config_path, connect_to_os=not args.without_os)
