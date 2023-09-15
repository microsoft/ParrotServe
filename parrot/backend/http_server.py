"""A fake server for testing."""

import argparse
import asyncio
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from uvicorn import Config, Server

from .engine import ExecutionEngine
from .backend_jobs import Fill, Generation
from ..utils import get_logger, run_coroutine_in_loop
from ..protocol.sampling_params import SamplingParams


logger = get_logger("Backend Server")

# FastAPI app
app = FastAPI()

# Engine
execution_engine: Optional[ExecutionEngine] = None


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
    await fill_job.finish_event.wait()

    return {
        "num_filled_tokens": len(fill_job.token_ids),
    }


@app.post("/generate")
async def generate(request: Request):
    payload = await request.json()
    session_id = payload["session_id"]
    context_id = payload["context_id"]
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
    num_freed_tokens = execution_engine.free_context(payload["context_id"])

    return {
        "num_freed_tokens": num_freed_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parrot backend HTTP server.")

    parser.add_argument("--engine_name", type=str, default="local_engine_0")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)

    args = parser.parse_args()

    # uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    execution_engine = ExecutionEngine(engine_name=args.engine_name)

    loop = asyncio.new_event_loop()
    config = Config(
        app=app,
        loop=loop,
        host="localhost",
        port=args.port,
        log_level="info",
    )
    uvicorn_server = Server(config)
    run_coroutine_in_loop(execution_engine.execute_loop(), loop=loop)
    loop.run_until_complete(uvicorn_server.serve())
