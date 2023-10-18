"""A fake server for testing."""

import argparse
import asyncio
from typing import Optional
from fastapi import FastAPI, Request
from uvicorn import Config, Server

from parrot.program.function import SemanticCall
from parrot.os.pcore import PCore
from parrot.os.engine import EngineRuntimeInfo
from parrot.utils import get_logger, create_task_in_loop


logger = get_logger("OS Server")

# FastAPI app
app = FastAPI()

# Engine
pcore: Optional[PCore] = None


@app.post("/vm_heartbeat")
async def vm_heartbeat(request: Request):
    pid = (await request.json())["pid"]
    pcore.vm_heartbeat(pid)


@app.post("/register_vm")
async def register_vm(request: Request):
    allocated_pid = pcore.register_vm()
    return {"pid": allocated_pid}


@app.post("/submit_call")
async def submit_call(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    call = SemanticCall.unpickle(payload["call"])
    pcore.submit_call(pid, call)
    return {}


@app.post("/placeholder_fetch")
async def placeholder_fetch(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    placeholder_id = payload["placeholder_id"]
    return {"content": await pcore.placeholder_fetch(pid, placeholder_id)}


@app.post("/engine_heartbeat")
async def engine_heartbeat(request: Request):
    payload = await request.json()
    engine_id = payload["engine_id"]
    engine_info = EngineRuntimeInfo(**payload["engine_info"])
    pcore.engine_heartbeat(engine_id, engine_info)
    return {}


@app.post("/register_engine")
async def register_engine(request: Request):
    payload = await request.json()
    name = payload["name"]
    host = payload["host"]
    port = payload["port"]
    engine_id = pcore.register_engine(name, host, port)
    return {"engine_id": engine_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parrot OS server.")

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file of PCore.",
        required=True,
    )

    args = parser.parse_args()
    pcore = PCore(args.config_path)

    loop = asyncio.new_event_loop()
    config = Config(
        app=app,
        loop=loop,
        host=pcore.os_config.host,
        port=pcore.os_config.port,
        log_level="info",
    )
    uvicorn_server = Server(config)
    # NOTE(chaofan): We use `fail_fast` because this project is still in development
    # For real deployment, maybe we don't need to quit the backend when there is an error
    create_task_in_loop(pcore.os_loop(), loop=loop, fail_fast=True)
    loop.run_until_complete(uvicorn_server.serve())
