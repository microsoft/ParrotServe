from typing import Type, List
import requests
import aiohttp
import dataclasses

from ..utils import get_logger
from .local_apis_response import *
from .sampling_params import SamplingParams
from ..constants import NONE_SESSION_ID


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
    for _ in range(retry_times):
        try:
            try:
                resp = requests.post(url, json=kwargs)
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


def fill(
    http_addr: str, context_id: int, parent_context_id: int, token_ids: List[int]
) -> FillResponse:
    try:
        return send_http_request(
            FillResponse,
            http_addr,
            "/fill",
            retry_times=1,
            session_id=NONE_SESSION_ID,  # No session id for simple fill
            context_id=context_id,
            parent_context_id=parent_context_id,
            token_ids=token_ids,
        )
    except BaseException as e:
        logger.error(f"Prefix init error in {http_addr} error: {e}")
        raise e


async def afill(
    http_addr: str,
    session_id: int,
    token_ids: List[int],
    context_id: int,
    parent_context_id: int,
) -> FillResponse:
    try:
        logger.debug(
            f"Send fill request to {http_addr}. Payload: "
            f"session_id={session_id}, "
            f"token_ids={token_ids}, "
            f"context_id={context_id}, "
            f"parent_context_id={parent_context_id}"
        )

        async with aiohttp.ClientSession() as client_session:
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
    except BaseException as e:
        logger.error(f"Fill error in {http_addr} error: {e}")
        raise e


async def agenerate(
    http_addr: str,
    session_id: int,
    context_id: int,
    parent_context_id: int,
    sampling_params: SamplingParams,
):
    try:
        async with aiohttp.ClientSession() as client_session:
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
    except BaseException as e:
        logger.error(f"Generate error in {http_addr} error: {e}")
        raise e


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
        logger.error(f"Free context error in {http_addr} error: {e}")
        raise e
