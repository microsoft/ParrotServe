# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Type
from aiohttp import ClientResponse
from pydantic import BaseModel
from requests import Response


"""
Use Pydantic to build response models.
"""


class BaseResponse(BaseModel):
    pass


def make_response(resp_cls: Type[BaseResponse], resp: Response):
    resp_data = resp.json()
    # init_data = [(field, resp_data[field]) for field in resp_cls.__fields__]
    init_data = [(field, resp_data[field]) for field in resp_data.keys()]
    return resp_cls(**dict(init_data))


async def async_make_response(resp_cls: Type[BaseResponse], resp: ClientResponse):
    resp_data = await resp.json()
    init_data = [(field, resp_data[field]) for field in resp_cls.__fields__]
    return resp_cls(**dict(init_data))
