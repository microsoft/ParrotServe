# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import asyncio
from dataclasses import asdict
from typing import Optional, Dict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server

from parrot.utils import (
    get_logger,
    create_task_in_loop,
    set_log_output_file,
    redirect_stdout_stderr_to_file,
)

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
    return await llm_engine.free_context(payload)


@app.post("/ping")
async def ping(request: Request):
    rt_info = llm_engine.get_runtime_info(profile=False)  # For speed
    return {
        "runtime_info": asdict(rt_info),
    }


def start_server(
    engine_config_path: str,
    connect_to_os: bool = True,
    override_args: Dict = {},
):
    global llm_engine
    global app

    llm_engine = create_engine(
        engine_config_path=engine_config_path,
        connect_to_os=connect_to_os,
        override_args=override_args,
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
    parser = argparse.ArgumentParser(description="Parrot engine HTTP server")

    parser.add_argument(
        "--host",
        type=str,
        help="Host of engine server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port of engine server.",
    )

    parser.add_argument(
        "--engine_name",
        type=str,
        help="Name of engine server.",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device of engine server.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file of the engine.",
        required=True,
    )

    parser.add_argument(
        "--without_os",
        action="store_true",
        help="Whether to start the engine without connecting to OS.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Path to the log directory. If not set, logs will be printed to stdout.",
    )

    parser.add_argument(
        "--log_filename",
        type=str,
        default="engine.log",
        help="Filename of the Engine server.",
    )

    parser.add_argument(
        "--release_mode",
        action="store_true",
        help="Run in release mode. In debug mode, Engine will print lots of logs.",
    )

    args = parser.parse_args()
    release_mode = args.release_mode

    if release_mode:
        # Disable logging
        import logging

        # We don't disable the error log
        logging.disable(logging.DEBUG)
        logging.disable(logging.INFO)

    # Set the log file
    if args.log_dir is not None:
        set_log_output_file(
            log_file_dir_path=args.log_dir,
            log_file_name=args.log_filename,
        )

        redirect_stdout_stderr_to_file(
            log_file_dir_path=args.log_dir,
            file_name="engine_stdout.out",
        )

    override_args = {}
    if args.host is not None:
        override_args["host"] = args.host
    if args.port is not None:
        override_args["port"] = args.port
    if args.engine_name is not None:
        override_args["engine_name"] = args.engine_name
    if args.device is not None:
        override_args["device"] = args.device

    # uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    start_server(
        engine_config_path=args.config_path,
        connect_to_os=not args.without_os,
        override_args=override_args,
    )
