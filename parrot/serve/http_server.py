# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import asyncio
import logging
import secrets
import traceback
from typing import Optional, TYPE_CHECKING
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from uvicorn import Config, Server
import os

from parrot.protocol.public.api_version import API_VERSION
from parrot.exceptions import ParrotCoreUserError, ParrotCoreInternalError

if TYPE_CHECKING:
    from parrot.serve.core import ParrotServeCore


logger = logging.getLogger("Parrot ServeCore Server")

# FastAPI app
app = FastAPI()

# Core
pcore: Optional["ParrotServeCore"] = None

# Mode
release_mode = False

AUTH_TOKEN_ENV = "PARROT_API_KEY"


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": message, "traceback": ""},
    )


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    expected_token = os.environ.get(AUTH_TOKEN_ENV)
    if not expected_token:
        return _error_response(
            503,
            f"ServeCore is locked because {AUTH_TOKEN_ENV} is not configured.",
        )

    authorization = request.headers.get("Authorization", "")
    scheme, _, supplied_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        supplied_token.encode("utf-8"), expected_token.encode("utf-8")
    ):
        response = _error_response(401, "Missing or invalid bearer token.")
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    return await call_next(request)


@app.exception_handler(ParrotCoreUserError)
async def parrot_core_internal_error_handler(
    request: Request, exc: ParrotCoreUserError
):
    traceback_info = "" if release_mode else traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": repr(exc),
            "traceback": traceback_info,
        },
    )


@app.exception_handler(ParrotCoreInternalError)
async def parrot_core_internal_error_handler(
    request: Request, exc: ParrotCoreInternalError
):
    raise exc


"""
Public APIs.
"""


@app.post(f"/{API_VERSION}/session")
async def register_session(request: Request):
    payload = await request.json()
    response = pcore.register_session(payload)
    return response


@app.delete(f"/{API_VERSION}" + "/session/{session_id}")
async def remove_session(session_id: int, request: Request):
    payload = await request.json()
    response = pcore.remove_session(session_id, payload)
    return response


@app.get(f"/{API_VERSION}" + "/session/{session_id}")
async def get_session_info(session_id: int, request: Request):
    raise NotImplementedError("Not implemented yet.")


@app.post(f"/{API_VERSION}/semantic_call")
async def submit_semantic_call(request: Request):
    # Sleep simulate network latency
    latency_open = os.environ.get("SIMULATE_NETWORK_LATENCY_PRT", None)
    assert (
        latency_open is not None
    ), "Please specify the environment variable SIMULATE_NETWORK_LATENCY_PRT"
    try:
        latency_open = int(latency_open)
        assert latency_open == 0 or latency_open == 1
    except:
        return ValueError("SIMULATE_NETWORK_LATENCY must 0/1.")

    # RTT
    if latency_open == 1:
        from parrot.testing.latency_simulator import get_latency

        latency = get_latency()
        await asyncio.sleep(latency / 2)

    payload = await request.json()
    response = pcore.submit_semantic_call(payload)

    if latency_open == 1:
        await asyncio.sleep(latency / 2)

    return response


@app.post(f"/{API_VERSION}/py_native_call")
async def submit_py_native_call(request: Request):
    return _error_response(
        410,
        "Python native calls are disabled because they execute client-supplied code.",
    )


@app.post(f"/{API_VERSION}/semantic_var")
async def register_semantic_variable(request: Request):
    payload = await request.json()
    response = pcore.register_semantic_variable(payload)
    return response


@app.post(f"/{API_VERSION}" + "/semantic_var/{var_id}")
async def set_semantic_variable(var_id: str, request: Request):
    payload = await request.json()
    response = pcore.set_semantic_variable(var_id, payload)
    return response


@app.get(f"/{API_VERSION}" + "/semantic_var/{var_id}")
async def get_semantic_variable(var_id: str, request: Request):
    payload = await request.json()
    response = await pcore.get_semantic_variable(var_id, payload)
    return response


@app.get(f"/{API_VERSION}/semantic_var")
async def get_semantic_variable_list(request: Request):
    raise NotImplementedError("Not implemented yet.")


"""
Internal APIs.
"""


@app.post("/engine_heartbeat")
async def engine_heartbeat(request: Request):
    return _error_response(
        403,
        "Dynamic engine registration and heartbeats are disabled by this hotfix.",
    )


@app.post("/register_engine")
async def register_engine(request: Request):
    return _error_response(
        403,
        "Dynamic engine registration and heartbeats are disabled by this hotfix.",
    )


def start_server(
    core_config_path: str,
    release_mode: bool = False,
    override_args: dict = {},
):
    global pcore
    global app

    if not os.environ.get(AUTH_TOKEN_ENV):
        raise RuntimeError(
            f"{AUTH_TOKEN_ENV} must be set before starting ServeCore."
        )

    from parrot.serve.core import create_serve_core
    from parrot.utils.async_utils import create_task_in_loop

    # Create ServeCore
    pcore = create_serve_core(
        core_config_path=core_config_path,
        release_mode=release_mode,
        override_args=override_args,
    )

    loop = asyncio.new_event_loop()
    config = Config(
        app=app,
        loop=loop,
        host=pcore.config.host,
        port=pcore.config.port,
        log_level="info",
    )
    uvicorn_server = Server(config)
    # NOTE(chaofan): We use `fail_fast` because this project is still in development
    # For real deployment, maybe we don't need to quit the backend when there is an error
    create_task_in_loop(pcore.serve_loop(), loop=loop, fail_fast=True)
    loop.run_until_complete(uvicorn_server.serve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parrot ServeCore http server")

    parser.add_argument(
        "--host",
        type=str,
        help="Host of PCore server",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="Port of PCore server",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the config file of PCore.",
        required=True,
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
        default="core.log",
        help="Filename of the OS server.",
    )

    parser.add_argument(
        "--release_mode",
        action="store_true",
        help="Run in release mode. In debug mode, "
        "Core will print logs and expose extra information to clients.",
    )

    args = parser.parse_args()
    release_mode = args.release_mode

    if release_mode:
        # Disable logging
        # We don't disable the error log
        logging.disable(logging.DEBUG)
        logging.disable(logging.INFO)

    # Set log output file
    if args.log_dir is not None:
        from parrot.utils.logging import set_log_output_file
        from parrot.utils.misc import redirect_stdout_stderr_to_file

        set_log_output_file(
            log_file_dir_path=args.log_dir,
            log_file_name=args.log_filename,
        )

        redirect_stdout_stderr_to_file(
            log_file_dir_path=args.log_dir,
            file_name="core_stdout.out",
        )

    override_args = {}
    if args.host is not None:
        override_args["host"] = args.host
    if args.port is not None:
        override_args["port"] = args.port

    start_server(
        core_config_path=args.config_path,
        release_mode=release_mode,
        override_args=override_args,
    )
