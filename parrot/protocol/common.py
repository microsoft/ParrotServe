from typing import Type
import requests
import aiohttp

from parrot.utils import get_logger

from .responses import BaseResponse, make_response, async_make_response


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
