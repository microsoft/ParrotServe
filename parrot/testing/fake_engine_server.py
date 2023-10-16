"""A fake server for testing."""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import time
import numpy as np

from ..utils import get_logger

# ---------- Constants ----------
TESTING_RANDOM_SEED = 2333
TESTING_SERVER_HOST = "localhost"
TESTING_SERVER_PORT = 8888
TESTING_FILL_PERTOKEN_TIME = 0.1
TESTING_DECODE_PERTOKEN_TIME = 0.1


app = FastAPI()


@app.post("/vm_heartbeat")
async def heartbeat(request: Request):
    return {
        "available_models": ["llama", "gpt2"],
    }


@app.post("/fill")
async def fill(request: Request):
    payload = await request.json()

    token_ids = payload["token_ids"]
    text = payload["text"]

    # Suppose the server will always fill all tokens
    # Simulate the time of filling tokens
    if token_ids is not None:
        length = len(token_ids)
    else:
        assert text is not None
        length = len(text.split())

    time.sleep(TESTING_FILL_PERTOKEN_TIME * length)

    return {
        "num_filled_tokens": length,
    }


@app.post("/generate")
async def generate(request: Request):
    global num_running_jobs
    global num_cached_tokens

    num_running_jobs += 1
    payload = await request.json()

    gen_len = int(np.random.exponential(32) + 3)
    # gen_len = 512
    gen_data = np.random.randint(10, 10000, size=(gen_len,)).tolist()

    num_cached_tokens += gen_len
    assert payload["context_id"] in context
    context[payload["context_id"]] += gen_len

    def generator():
        for data in gen_data:
            # Simulate the time of decoding tokens
            time.sleep(TESTING_DECODE_PERTOKEN_TIME)
            yield data.to_bytes(4, "big")

    num_running_jobs -= 1

    return StreamingResponse(generator())


@app.post("/free_context")
async def free_context(request: Request):
    global num_cached_tokens

    payload = await request.json()
    # assert payload["context_id"] in context

    num_freed_tokens = 0

    if payload["context_id"] in context:
        num_cached_tokens -= context[payload["context_id"]]
        num_freed_tokens = context[payload["context_id"]]

    return {
        "num_freed_tokens": num_freed_tokens,
    }


if __name__ == "__main__":
    np.random.seed(TESTING_RANDOM_SEED)
    uvicorn.run(
        app, host=TESTING_SERVER_HOST, port=TESTING_SERVER_PORT, log_level="info"
    )
