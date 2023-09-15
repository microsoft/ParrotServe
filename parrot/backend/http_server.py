"""A fake server for testing."""

import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from .engine import ExecutionEngine

from .backend_jobs import Fill, Generation
from ..utils import get_logger
from ..protocol.sampling_params import SamplingParams


logger = get_logger("Backend Server")

# FastAPI app
app = FastAPI()

# Engine
execution_engine = ExecutionEngine()


@app.post("/heartbeat")
async def heartbeat(request: Request):
    return execution_engine.stats()


@app.post("/fill")
async def fill(request: Request):
    payload = await request.json()
    fill_job = Fill(
        session_id=payload["session_id"],
        context_id=payload["context_id"],
        parent_context_id=payload["parent_context_id"],
        token_ids=payload["token_ids"],
    )
    execution_engine.add_job(fill_job)
    await fill_job.finished.wait()

    return {
        "filled_tokens_num": len(fill_job.token_ids),
    }


@app.post("/generate")
async def generate(request: Request):
    payload = await request.json()
    session_id = (payload["session_id"],)
    context_id = (payload["context_id"],)
    payload.pop("session_id")
    payload.pop("context_id")

    generation_job = Generation(
        session_id=session_id,
        context_id=context_id,
        sampling_params=SamplingParams(**payload),
    )
    execution_engine.add_job(generation_job)

    return StreamingResponse(generation_job.generator())


@app.post("/free_context")
async def free_context(request: Request):
    payload = await request.json()
    free_tokens_num = execution_engine.free_context(payload["context_id"])

    return {
        "free_tokens_num": free_tokens_num,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parrot backend HTTP server.")

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
