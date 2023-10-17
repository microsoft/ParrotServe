"""A fake server for testing."""

from fastapi import FastAPI, Request
import uvicorn
import numpy as np

from parrot.program.function import SemanticCall
from parrot.utils import get_logger

# ---------- Constants ----------
TESTING_RANDOM_SEED = 2333
TESTING_SERVER_HOST = "localhost"
TESTING_SERVER_PORT = 8888


logger = get_logger("Fake OS Server")


app = FastAPI()


@app.post("/vm_heartbeat")
async def heartbeat(request: Request):
    pid = (await request.json())["pid"]
    logger.info(f"Received heartbeat from VM (pid: {pid}).")
    return {
        "mem_used": 0.0,
        "mem_threads": 0,
    }


@app.post("/register_vm")
async def register_vm(request: Request):
    allocated_pid = 0
    logger.info(f"Register VM. Allocated pid: {allocated_pid}.")
    return {"pid": allocated_pid}


@app.post("/submit_call")
async def submit_call(request: Request):
    payload = await request.json()
    pid = payload["pid"]
    call = SemanticCall.unpickle(payload["call"])
    logger.info(f"Execute function {call.func.name} in VM (pid: {pid}).")
    return {}


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
