# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""A fake server for testing."""

from typing import Dict
from fastapi import FastAPI, Request
import uvicorn
import numpy as np

from parrot.constants import DEFAULT_SERVER_HOST, DEFAULT_CORE_SERVER_PORT
from parrot.utils import get_logger
from parrot.protocol.public.api_version import API_VERSION

# ---------- Constants ----------
TESTING_RANDOM_SEED = 2333
TESTING_SERVER_HOST = DEFAULT_SERVER_HOST
TESTING_SERVER_PORT = DEFAULT_CORE_SERVER_PORT
TESTING_SERVER_URL = f"http://{TESTING_SERVER_HOST}:{TESTING_SERVER_PORT}"


logger = get_logger("Fake ServeCore Server")


app = FastAPI()

_sessions = set()
_session_counter = 0


@app.post(f"/{API_VERSION}/session")
async def register_session(request: Request):
    global _sessions
    global _session_counter

    logger.debug(f"Register session.")
    session_id = _session_counter
    _sessions.add(session_id)
    _session_counter += 1
    return {"session_id": session_id, "session_auth": "1"}


@app.delete(f"/{API_VERSION}" + "/session/{session_id}")
async def remove_session(session_id: int, request: Request):
    global _sessions

    logger.debug(f"Remove session id={session_id}.")
    payload = await request.json()
    assert session_id in _sessions
    _sessions.remove(session_id)
    return {}


_request_counter = 0


@app.post(f"/{API_VERSION}/submit_semantic_call")
async def submit_semantic_call(request: Request):
    global _request_counter

    payload = await request.json()

    session_id = payload["session_id"]
    request_id = _request_counter
    _request_counter += 1

    logger.debug(
        f"Submit semantic call. Session id={session_id}. Request id={request_id}."
    )
    return {
        "request_id": request_id,
        "created_vars": [],
    }


_semantic_vars: Dict[str, str] = {}
_var_counter = 0


@app.post(f"/{API_VERSION}/semantic_var")
async def register_semantic_variable(request: Request):
    global _semantic_vars
    global _var_counter

    payload = await request.json()
    name = payload["var_name"]
    logger.debug(f"Register semantic variable {name}.")
    var_id = str(_var_counter)
    _var_counter += 1
    _semantic_vars[var_id] = ""
    return {
        "var_id": var_id,
    }


@app.post(f"/{API_VERSION}" + "/semantic_var/{var_id}")
async def set_semantic_variable(var_id: str, request: Request):
    payload = await request.json()
    content = payload["content"]
    logger.debug(f"Set semantic variable {var_id}. Content: {content}.")
    assert var_id in _semantic_vars
    _semantic_vars[var_id] = content
    return {}


@app.get(f"/{API_VERSION}" + "/semantic_var/{var_id}")
async def get_semantic_variable(var_id: str, request: Request):
    payload = await request.json()
    content = _semantic_vars[var_id]
    logger.debug(f"Get semantic variable {var_id}. Content: {content}.")
    return {"content": content}


@app.post("/engine_heartbeat")
async def engine_heartbeat(request: Request):
    payload = await request.json()
    engine_id = payload["engine_id"]
    engine_name = payload["engine_name"]
    logger.debug(f"Received heartbeat from Engine {engine_name} (id={engine_id}).")
    return {}


@app.post("/register_engine")
async def register_engine(request: Request):
    payload = await request.json()
    engine_name = payload["engine_config"]["engine_name"]
    allocated_engine_id = 0
    logger.debug(
        f"Register Engine {engine_name}. Allocated engine_id: {allocated_engine_id}."
    )
    return {"engine_id": allocated_engine_id}


if __name__ == "__main__":
    np.random.seed(TESTING_RANDOM_SEED)
    uvicorn.run(
        app,
        host=TESTING_SERVER_HOST,
        port=TESTING_SERVER_PORT,
        log_level="debug",
    )
