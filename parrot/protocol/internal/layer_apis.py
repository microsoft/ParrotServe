# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import aiohttp
from dataclasses import asdict
from typing import List, Dict

from parrot.utils import get_logger

from ..base_response import BaseResponse
from ..http_utils import send_http_request
from .runtime_info import EngineRuntimeInfo


"""
Internal APIs (used for communication between ServeLayer and EngineLayer).

Engine Management (Server side):
    - register_engine POST
    - engine_heartbeat POST

Engine Management (Engine side):
    - ping POST

Context & LLMs:
    - free_context POST
    - fill POST
    - generate POST
    - generate_stream POST
"""


logger = get_logger("Layer APIs")


# ---------- Responses ----------


class EngineHeartbeatResponse(BaseResponse):
    pass


class RegisterEngineResponse(BaseResponse):
    engine_id: int


class PingEngineResponse(BaseResponse):
    pong: bool = True
    runtime_info: Dict = {}


class FreeContextResponse(BaseResponse):
    context_len: int


class FillResponse(BaseResponse):
    filled_len: int


class GenerateResponse(BaseResponse):
    generated_text: str
    generated_ids: List[int]


# ---------- Serve Layer to Engine Layer APIs ----------


def free_context(http_addr: str, context_id: int) -> FreeContextResponse:
    try:
        return send_http_request(
            FreeContextResponse,
            http_addr,
            "/free_context",
            retry_times=1,
            context_id=context_id,
        )
    except BaseException as e:
        logger.error(f"Free context error in {http_addr}. Error: {e}")
        raise e


def ping_engine(http_addr: str) -> PingEngineResponse:
    try:
        return send_http_request(
            PingEngineResponse,
            http_addr,
            "/ping",
            retry_times=5,
        )
    except BaseException as e:
        print(e.args)
        return PingEngineResponse(pong=False)


# ---------- Engine Layer to Serve Layer APIs ----------


def register_engine(
    http_addr: str,
    engine_config: "EngineConfig",
) -> RegisterEngineResponse:
    try:
        return send_http_request(
            RegisterEngineResponse,
            http_addr,
            "/register_engine",
            retry_times=1,
            engine_config=asdict(engine_config),
        )
    except BaseException as e:
        logger.error(
            f"Register engine {engine_config.engine_name} error in {http_addr}. Error: {e}"
        )
        raise e


def engine_heartbeat(
    http_addr: str,
    engine_id: int,
    engine_name: str,
    runtime_info: EngineRuntimeInfo,
) -> EngineHeartbeatResponse:
    try:
        return send_http_request(
            response_cls=EngineHeartbeatResponse,
            http_addr=http_addr,
            api_url="/engine_heartbeat",
            retry_times=3,
            engine_id=engine_id,
            engine_name=engine_name,
            runtime_info=asdict(runtime_info),
        )
    except BaseException as e:
        logger.error(
            f"Check engine heartbeat error. Engine: {engine_name} (id={engine_id}), Error: {e}"
        )
        raise e
