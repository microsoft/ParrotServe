from fastapi import FastAPI, Request
from ..utils import get_logger

logger = get_logger("Backend Server")

app = FastAPI()


@app.get("/heartbeat")
async def heartbeat(request: Request):
    pass


@app.get("/fill")
async def fill(request: Request):
    pass


@app.get("/generate")
async def generate(request: Request):
    pass


@app.get("/free_context")
async def free_context(request: Request):
    pass
