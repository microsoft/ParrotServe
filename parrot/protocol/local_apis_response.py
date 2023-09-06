from typing import List, Type
from pydantic import BaseModel
from requests import Response


class BaseResponse(BaseModel):
    pass


class HeartbeatResponse(BaseResponse):
    model_ready: bool
    cached_tokens: int
    running_jobs: int


class FillResponse(BaseResponse):
    filled_tokens_num: int


class GenerationResponse(BaseResponse):
    gen_tokens: List[int]
    finish_type: int


class FreeContextResponse(BaseResponse):
    free_tokens_num: int


def make_response(resp_cls: Type[BaseResponse], resp: Response):
    resp_data = resp.json()
    init_data = [(field, resp_data[field]) for field in resp_cls.__fields__]
    return resp_cls(**dict(init_data))
