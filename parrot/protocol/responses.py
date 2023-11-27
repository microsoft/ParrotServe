# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Dict, Type
from pydantic import BaseModel
from requests import Response
from aiohttp import ClientResponse


class BaseResponse(BaseModel):
    pass


class FillResponse(BaseResponse):
    filled_len: int


class GenerateResponse(BaseResponse):
    generated_text: str
    generated_ids: List[int]


class VMHeartbeatResponse(BaseResponse):
    mem_used: float
    num_threads: int


class RegisterVMResponse(BaseResponse):
    pid: int


class SubmitCallResponse(BaseResponse):
    pass


class PlaceholderSetResponse(BaseResponse):
    pass

class PlaceholderFetchResponse(BaseResponse):
    content: str


class EngineHeartbeatResponse(BaseResponse):
    pass


class RegisterEngineResponse(BaseResponse):
    engine_id: int


class FreeContextResponse(BaseResponse):
    context_len: int


class PingResponse(BaseResponse):
    pong: bool = True
    runtime_info: Dict = {}


def make_response(resp_cls: Type[BaseResponse], resp: Response):
    resp_data = resp.json()
    # init_data = [(field, resp_data[field]) for field in resp_cls.__fields__]
    init_data = [(field, resp_data[field]) for field in resp_data.keys()]
    return resp_cls(**dict(init_data))


async def async_make_response(resp_cls: Type[BaseResponse], resp: ClientResponse):
    resp_data = await resp.json()
    init_data = [(field, resp_data[field]) for field in resp_cls.__fields__]
    return resp_cls(**dict(init_data))
