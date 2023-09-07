from typing import Type, List
import requests
import aiohttp
import dataclasses
from requests.exceptions import RequestException
from pydantic import ValidationError

from ..utils import get_logger
from .local_apis_response import *
from .sampling_params import SamplingParams


logger = get_logger("API")


def send_http_request(
    response_cls: Type[BaseResponse],
    http_addr: str,
    api_url: str,
    retry_times: int,
    **kwargs,
) -> BaseResponse:
    url = http_addr + api_url
    error = None
    timeout = 3
    for _ in range(retry_times):
        try:
            try:
                resp = requests.post(url, json=kwargs, timeout=timeout)
                assert resp.status_code == 200
                return make_response(response_cls, resp)
            except BaseException as e:
                raise e
        except BaseException as e:
            error = e

    assert error is not None
    # forward to caller side
    raise error


async def async_send_http_request(
    client_session: aiohttp.ClientSession,
    response_cls: Type[BaseResponse],
    http_addr: str,
    api_url: str,
    **kwargs,
) -> BaseResponse:
    url = http_addr + api_url
    async with client_session.post(url, json=kwargs) as resp:
        assert resp.ok, "Send http request error."
        return await async_make_response(response_cls, resp)


async def async_send_http_request_streaming(
    client_session: aiohttp.ClientSession,
    http_addr: str,
    api_url: str,
    **kwargs,
):
    url = http_addr + api_url
    async with client_session.post(url, json=kwargs) as reader:
        # assert resp.ok, "Send http request error."
        async for chunk in reader.content.iter_chunked(4):
            yield int().from_bytes(chunk, "big")


def check_heartbeat(engine_name: str, http_addr: str) -> HeartbeatResponse:
    try:
        return send_http_request(
            HeartbeatResponse, http_addr, "/heartbeat", retry_times=3
        )
    except BaseException as e:
        logger.error(f"Check heartbeat error in {engine_name} error: {e}")
        raise e


def prefix_init(http_addr: str, context_id: int, token_ids: List[int]) -> FillResponse:
    try:
        return send_http_request(
            FillResponse,
            http_addr,
            "/fill",
            retry_times=1,
            session_id=-1,  # No session id for prefix init
            context_id=context_id,
            parent_context_id=-1,  # Since we are init a new prefix context
            token_ids=token_ids,
        )
    except BaseException as e:
        logger.error(f"Prefix init error in {http_addr} error: {e}")
        raise e


async def fill(
    client_session: aiohttp.ClientSession,
    http_addr: str,
    session_id: int,
    context_id: int,
    parent_context_id: int,
    token_ids: List[int],
) -> FillResponse:
    return await async_send_http_request(
        client_session,
        FillResponse,
        http_addr,
        "/fill",
        session_id=session_id,
        context_id=context_id,
        parent_context_id=parent_context_id,
        token_ids=token_ids,
    )


async def generate(
    client_session: aiohttp.ClientSession,
    http_addr: str,
    session_id: int,
    context_id: int,
    parent_context_id: int,
    sampling_params: SamplingParams,
):
    async for resp in async_send_http_request_streaming(
        client_session,
        http_addr,
        "/generate",
        session_id=session_id,
        context_id=context_id,
        parent_context_id=parent_context_id,
        **dataclasses.asdict(sampling_params),
    ):
        yield resp


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
        raise e
