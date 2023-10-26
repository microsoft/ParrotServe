"""A fake server for testing."""

import argparse
import asyncio
import traceback
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from uvicorn import Config, Server

from parrot.program.function import SemanticCall
from parrot.os.pcore import PCore
from parrot.os.engine import EngineRuntimeInfo
from parrot.engine.config import EngineConfig
from parrot.utils import get_logger, create_task_in_loop
from parrot.exceptions import ParrotOSUserError, ParrotOSInteralError


logger = get_logger("OS Server")

# FastAPI app
app = FastAPI()

# Engine
pcore: Optional[PCore] = None

# Mode
release_mode = False


@app.exception_handler(ParrotOSUserError)
async def parrot_os_internal_error_handler(request: Request, exc: ParrotOSUserError):
    traceback_info = "" if release_mode else traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": repr(exc),
            "traceback": traceback_info,
        },
    )


@app.exception_handler(ParrotOSInteralError)
async def parrot_os_internal_error_handler(request: Request, exc: ParrotOSInteralError):
    raise exc


@app.post("/vm_heartbeat")
async def vm_heartbeat(request: Request):
    pid = (await request.json())["pid"]
    logger.debug(f"VM heartbeat received: pid={pid}")
    return pcore.vm_heartbeat(pid)


@app.post("/register_vm")
async def register_vm(request: Request):
    logger.debug(f"Register VM received.")
    allocated_pid = pcore.register_vm()
    return {"pid": allocated_pid}


@app.post("/submit_call")
async def submit_call(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    logger.debug(f"Submit call received: pid={pid}")
    call = SemanticCall.unpickle(payload["call"])
    pcore.submit_call(pid, call)
    return {}


@app.post("/placeholder_fetch")
async def placeholder_fetch(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    placeholder_id = payload["placeholder_id"]
    logger.debug(
        f"Placeholder fetch received: pid={pid}, placeholder_id={placeholder_id}"
    )
    return {"content": await pcore.placeholder_fetch(pid, placeholder_id)}


@app.post("/engine_heartbeat")
async def engine_heartbeat(request: Request):
    payload = await request.json()
    engine_id = payload["engine_id"]
    logger.debug(f"Engine heartbeat received: id={engine_id}.")
    engine_info = EngineRuntimeInfo(**payload["runtime_info"])
    pcore.engine_heartbeat(engine_id, engine_info)
    return {}


@app.post("/register_engine")
async def register_engine(request: Request):
    payload = await request.json()
    logger.debug(f"Register engine received.")
    engine_config = EngineConfig(**payload["engine_config"])
    engine_id = pcore.register_engine(engine_config)
    return {"engine_id": engine_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parrot OS server")

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file of PCore.",
        required=True,
    )

    parser.add_argument(
        "--release-mode",
        action="store_true",
        help="Run in release mode. In debug mode, "
        "OS will print more logs and expose extra information to clients.",
    )

    args = parser.parse_args()
    release_mode = args.release_mode
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
