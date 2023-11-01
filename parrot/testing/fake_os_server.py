"""A fake server for testing."""

from fastapi import FastAPI, Request
import uvicorn
import numpy as np

from parrot.program.function import SemanticCall
from parrot.constants import DEFAULT_SERVER_HOST, DEFAULT_OS_SERVER_PORT
from parrot.utils import get_logger

# ---------- Constants ----------
TESTING_RANDOM_SEED = 2333
TESTING_SERVER_HOST = DEFAULT_SERVER_HOST
TESTING_SERVER_PORT = DEFAULT_OS_SERVER_PORT


logger = get_logger("Fake OS Server")


app = FastAPI()


@app.post("/vm_heartbeat")
async def vm_heartbeat(request: Request):
    pid = (await request.json())["pid"]
    logger.info(f"Received heartbeat from VM (pid: {pid}).")
    return {
        "mem_used": 0.0,
        "num_threads": 0,
    }


@app.post("/register_vm")
async def register_vm(request: Request):
    allocated_pid = 0
    logger.info(f"Register VM. Allocated pid: {allocated_pid}.")
    return {"pid": allocated_pid}


@app.post("/engine_heartbeat")
async def engine_heartbeat(request: Request):
    payload = await request.json()
    engine_id = payload["engine_id"]
    engine_name = payload["engine_name"]
    logger.info(f"Received heartbeat from Engine {engine_name} (id={engine_id}).")
    return {}


@app.post("/register_engine")
async def register_engine(request: Request):
    payload = await request.json()
    engine_name = payload["engine_config"]["engine_name"]
    allocated_engine_id = 0
    logger.info(
        f"Register Engine {engine_name}. Allocated engine_id: {allocated_engine_id}."
    )
    return {"engine_id": allocated_engine_id}


@app.post("/submit_call")
async def submit_call(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    call = SemanticCall.unpickle(payload["call"])
    logger.info(f"Execute function {call.func.name} in VM (pid: {pid}).")
    return {
        "context_id": 0,
    }


@app.post("/placeholder_fetch")
async def placeholder_fetch(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    placeholder_id = payload["placeholder_id"]
    logger.info(f"Fetch placeholder {placeholder_id} in VM (pid: {pid}).")
    return {"content": "placeholder_xxx"}


if __name__ == "__main__":
    np.random.seed(TESTING_RANDOM_SEED)
    uvicorn.run(
        app, host=TESTING_SERVER_HOST, port=TESTING_SERVER_PORT, log_level="info"
    )
