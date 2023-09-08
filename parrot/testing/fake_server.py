from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import time
import numpy as np

from parrot.utils import get_logger

logger = get_logger("Backend Server")

app = FastAPI()


# Constants

FILL_PERTOKEN_TIME = 0.01
DECODE_PERTOKEN_TIME = 0.001


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
    # Simulate the time of filling tokens
    time.sleep(FILL_PERTOKEN_TIME * len(payload["token_ids"]))
    return {
        "filled_tokens_num": len(payload["token_ids"]),
    }


@app.post("/generate")
async def generate(request: Request):
    gen_len = int(np.random.exponential(32) + 3)
    gen_data = np.random.randint(10, 10000, size=(gen_len,)).tolist()

    def generator():
        for data in gen_data:
            # Simulate the time of decoding tokens
            time.sleep(DECODE_PERTOKEN_TIME)
            yield data.to_bytes(4, "big")

    return StreamingResponse(generator())


@app.post("/free_context")
async def free_context(request: Request):
    return {
        "free_tokens_num": 0,
    }


if __name__ == "__main__":
    np.random.seed(2333)
    uvicorn.run(app, host="localhost", port=8888, log_level="info")
