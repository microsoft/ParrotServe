# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass, asdict
from typing import List, Optional, AsyncGenerator
import time
import aiohttp

from parrot.utils import get_logger, time_counter_in_nanoseconds

from ..http_utils import (
    send_http_request,
    async_send_http_request,
    async_send_http_request_streaming,
    logger,
)
from ...sampling_config import SamplingConfig
from .layer_apis import FillResponse, GenerateResponse


logger = get_logger("Primitive")


@dataclass
class Primitive:
    """Base class for LLM primitives."""

    session_id: int
    task_id: int
    context_id: int
    parent_context_id: int
    end_flag: bool


@dataclass
class Fill(Primitive):
    """Fill primitive is corresponding to the `prefill` stage in LLM.

    Its mission is to fill the KV cache in the execution engine, extending the context
    using the input tokens.
    """

    token_ids: Optional[List[int]] = None
    text: Optional[str] = None

    def post(self, engine_url: str) -> FillResponse:
        try:
            st = time_counter_in_nanoseconds()
            resp: FillResponse = send_http_request(
                response_cls=FillResponse,
                http_addr=engine_url,
                api_url="/fill",
                retry_times=1,
                session_id=self.session_id,
                task_id=self.task_id,
                context_id=self.context_id,
                parent_context_id=self.parent_context_id,
                end_flag=self.end_flag,
                token_ids=self.token_ids,
                text=self.text,
            )
            ed = time_counter_in_nanoseconds()
            logger.debug(
                f"Fill request latency: {(ed - st) / 1e6} ms. session_id={self.session_id}, task_id={self.task_id}"
            )
            return resp
        except BaseException as e:
            logger.error(f"Fill error in {engine_url} error: {e}")
            raise e

    async def apost(self, engine_url: str) -> FillResponse:
        try:
            async with aiohttp.ClientSession() as client_session:
                st = time_counter_in_nanoseconds()
                resp: FillResponse = await async_send_http_request(
                    client_session=client_session,
                    response_cls=FillResponse,
                    http_addr=engine_url,
                    api_url="/fill",
                    session_id=self.session_id,
                    task_id=self.task_id,
                    context_id=self.context_id,
                    end_flag=self.end_flag,
                    parent_context_id=self.parent_context_id,
                    token_ids=self.token_ids,
                    text=self.text,
                )
                ed = time_counter_in_nanoseconds()
                logger.debug(
                    f"Fill request latency: {(ed - st) / 1e6} ms. session_id={self.session_id}, task_id={self.task_id}"
                )
            # self.context.token_nums += resp.filled_len
            return resp
        except BaseException as e:
            logger.error(f"Fill error in {engine_url} error: {e}")
            raise e


@dataclass
class Generate(Primitive):
    """Generate primitive is corresponding to the `decode` stage in LLM.

    Its mission is to generate the output tokens based on certain context.
    """

    sampling_config: SamplingConfig

    async def apost(self, engine_url: str) -> GenerateResponse:
        try:
            async with aiohttp.ClientSession() as client_session:
                st = time_counter_in_nanoseconds()
                resp: GenerateResponse = await async_send_http_request(
                    client_session=client_session,
                    response_cls=GenerateResponse,
                    http_addr=engine_url,
                    api_url="/generate",
                    session_id=self.session_id,
                    task_id=self.task_id,
                    context_id=self.context_id,
                    parent_context_id=self.parent_context_id,
                    end_flag=self.end_flag,
                    sampling_config=asdict(self.sampling_config),
                )
                ed = time_counter_in_nanoseconds()
                logger.debug(
                    f"Generate request latency: {(ed - st) / 1e6} ms. session_id={self.session_id}, task_id={self.task_id}"
                )
                # self.context.token_nums += len(resp.generated_ids)
                return resp
        except BaseException as e:
            logger.error(f"Generate error in {engine_url} error: {e}")
            raise e

    async def astream(self, engine_url: str) -> AsyncGenerator:
        try:
            async with aiohttp.ClientSession() as client_session:
                st = time_counter_in_nanoseconds()
                async for resp in async_send_http_request_streaming(
                    client_session=client_session,
                    http_addr=engine_url,
                    api_url="/generate_stream",
                    session_id=self.session_id,
                    task_id=self.task_id,
                    context_id=self.context_id,
                    end_flag=self.end_flag,
                    parent_context_id=self.parent_context_id,
                    sampling_config=asdict(self.sampling_config),
                ):
                    # self.context.token_nums += 1
                    yield resp
                ed = time_counter_in_nanoseconds()
                logger.debug(
                    f"Generate stream latency: {(ed - st) / 1e6} ms. session_id={self.session_id}, task_id={self.task_id}"
                )
        except BaseException as e:
            logger.error(f"Generate error in {engine_url} error: {e}")
            raise e
