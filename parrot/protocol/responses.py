from typing import List, Type
from pydantic import BaseModel
from requests import Response
from aiohttp import ClientResponse


class BaseResponse(BaseModel):
    pass


class FillResponse(BaseResponse):
    num_filled_tokens: int


class VMHeartbeatResponse(BaseResponse):
    live: bool


class RegisterVMResponse(BaseResponse):
    pid: int


class ThreadStartResponse(BaseResponse):
    tokenizer: str
    interpret_type: str


class ThreadEndResponse(BaseResponse):
    num_freed_tokens: int


class EngineHeartbeatResponse(BaseResponse):
    pass


class RegisterEngineResponse(BaseResponse):
    engine_id: int


class FreeContextResponse(BaseResponse):
    num_freed_tokens: int


def make_response(resp_cls: Type[BaseResponse], resp: Response):
    resp_data = resp.json()
    init_data = [(field, resp_data[field]) for field in resp_cls.__fields__]
    return resp_cls(**dict(init_data))


async def async_make_response(resp_cls: Type[BaseResponse], resp: ClientResponse):
    resp_data = await resp.json()
    init_data = [(field, resp_data[field]) for field in resp_cls.__fields__]
    return resp_cls(**dict(init_data))
