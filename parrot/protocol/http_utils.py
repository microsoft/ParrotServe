# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Type, Optional, Literal
import requests
import aiohttp

from parrot.utils import get_logger

from .internal.responses import BaseResponse, make_response, async_make_response


logger = get_logger("API")


def send_http_request(
    response_cls: Type[BaseResponse],
    http_addr: str,
    api_url: str,
    retry_times: int,
    timeout: Optional[int] = None,
    method: Literal["GET", "POST", "DELETE"] = "POST",
    **kwargs,
) -> BaseResponse:
    url = http_addr + api_url
    error = None
    error_resp = None
    for _ in range(retry_times):
        try:
            if method == "GET":
                resp = requests.get(url, json=kwargs, timeout=timeout)
            elif method == "POST":
                resp = requests.post(url, json=kwargs, timeout=timeout)
            elif method == "DELETE":
                resp = requests.delete(url, json=kwargs, timeout=timeout)
            else:
                raise ValueError(f"Invalid http method: {method}")

            if resp.status_code != 200:
                error_resp = resp
                continue
            return make_response(response_cls, resp)
        except BaseException as e:
            error = e

    if error_resp is not None and error_resp.status_code == 500:
        resp_data = error_resp.json()
        assert "error" in resp_data
        assert "traceback" in resp_data
        raise RuntimeError(f"{resp_data['error']}\n{resp_data['traceback']}")

    assert error is not None
    # forward to caller side
    raise error


async def async_send_http_request(
    client_session: aiohttp.ClientSession,
    response_cls: Type[BaseResponse],
    http_addr: str,
    api_url: str,
    timeout=None,
    **kwargs,
) -> BaseResponse:
    url = http_addr + api_url
    async with client_session.post(url, json=kwargs, timeout=timeout) as resp:
        assert resp.ok, f"Send http request error: {resp.reason}"
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
