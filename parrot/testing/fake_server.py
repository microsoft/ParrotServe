from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import time

from parrot.utils import get_logger

logger = get_logger("Backend Server")

app = FastAPI()


@app.post("/heartbeat")
async def heartbeat(request: Request):
    return {
        "model_ready": True,
        "cached_tokens": 0,
        "running_jobs": 0,
    }


@app.post("/fill")
async def fill(request: Request):
    payload = await request.json()
    # Suppose the server will always fill all tokens
    return {
        "filled_tokens_num": len(payload["token_ids"]),
    }


@app.post("/generate")
async def generate(request: Request):
    def generator():
        data_list = [1, 2, 3, 4, 5]
        for data in data_list:
            time.sleep(0.1)
            yield data.to_bytes(4, "big")

    return StreamingResponse(generator())


@app.post("/free_context")
async def free_context(request: Request):
    return {
        "free_tokens_num": 0,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8888, log_level="info")
