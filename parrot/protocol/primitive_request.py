# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass, asdict
from typing import List, Optional
import time
import aiohttp

from parrot.constants import NONE_THREAD_ID
from parrot.utils import get_logger

from .common import (
    send_http_request,
    async_send_http_request,
    async_send_http_request_streaming,
    logger,
)
from .sampling_config import SamplingConfig
from .responses import FillResponse, GenerateResponse


logger = get_logger("Primitive")


@dataclass
class Primitive:
    """Base class for LLM primitives."""

    pid: int
    tid: int
    context: "Context"
    end_flag: bool


@dataclass
class Fill(Primitive):
    """Fill primitive is corresponding to the `prefill` stage in LLM.

    Its mission is to fill the KV cache in the execution engine, extending the context
    using the input tokens.
    """

    token_ids: Optional[List[int]] = None
    text: Optional[str] = None

    def post(self) -> FillResponse:
        # NOTE(chaofan): For simple fill, there is no thread id.
        # assert self.tid == NONE_THREAD_ID

        try:
            st = time.perf_counter_ns()
            resp: FillResponse = send_http_request(
                response_cls=FillResponse,
                http_addr=self.context.engine_url,
                api_url="/fill",
                retry_times=1,
                pid=self.pid,
                tid=self.tid,
                context_id=self.context.context_id,
                end_flag=self.end_flag,
                parent_context_id=self.context.parent_context_id,
                token_ids=self.token_ids,
                text=self.text,
            )
            ed = time.perf_counter_ns()
            logger.debug(
                f"Fill request latency: {(ed - st) / 1e6} ms. pid={self.pid}, tid={self.tid}"
            )
            self.context.token_nums += resp.filled_len
            return resp
        except BaseException as e:
            logger.error(f"Fill error in {self.context.engine_url} error: {e}")
            raise e

    async def apost(self) -> FillResponse:
        # NOTE: For thread fill, there is a thread id.
        assert self.tid != NONE_THREAD_ID

        try:
            async with aiohttp.ClientSession() as client_session:
                st = time.perf_counter_ns()
                resp: FillResponse = await async_send_http_request(
                    client_session=client_session,
                    response_cls=FillResponse,
                    http_addr=self.context.engine_url,
                    api_url="/fill",
                    pid=self.pid,
                    tid=self.tid,
                    context_id=self.context.context_id,
                    end_flag=self.end_flag,
                    parent_context_id=self.context.parent_context_id,
                    token_ids=self.token_ids,
                    text=self.text,
                )
                ed = time.perf_counter_ns()
                logger.debug(
                    f"Fill request latency: {(ed - st) / 1e6} ms. pid={self.pid}, tid={self.tid}"
                )
            self.context.token_nums += resp.filled_len
            return resp
        except BaseException as e:
            logger.error(f"Fill error in {self.context.engine_url} error: {e}")
            raise e


@dataclass
class Generate(Primitive):
    """Generate primitive is corresponding to the `decode` stage in LLM.

    Its mission is to generate the output tokens based on certain context.
    """

    sampling_config: SamplingConfig

    async def apost(self) -> GenerateResponse:
        try:
            async with aiohttp.ClientSession() as client_session:
                st = time.perf_counter_ns()
                resp: GenerateResponse = await async_send_http_request(
                    client_session=client_session,
                    response_cls=GenerateResponse,
                    http_addr=self.context.engine_url,
                    api_url="/generate",
                    pid=self.pid,
                    tid=self.tid,
                    context_id=self.context.context_id,
                    end_flag=self.end_flag,
                    parent_context_id=self.context.parent_context_id,
                    sampling_config=asdict(self.sampling_config),
                )
                ed = time.perf_counter_ns()
                logger.debug(
                    f"Generate request latency: {(ed - st) / 1e6} ms. pid={self.pid}, tid={self.tid}"
                )
                self.context.token_nums += len(resp.generated_ids)
                return resp
        except BaseException as e:
            logger.error(f"Generate error in {self.context.engine_url} error: {e}")
            raise e

    async def astream(self):
        try:
            async with aiohttp.ClientSession() as client_session:
                st = time.perf_counter_ns()
                async for resp in async_send_http_request_streaming(
                    client_session=client_session,
                    http_addr=self.context.engine_url,
                    api_url="/generate_stream",
                    pid=self.pid,
                    tid=self.tid,
                    context_id=self.context.context_id,
                    end_flag=self.end_flag,
                    parent_context_id=self.context.parent_context_id,
                    sampling_config=asdict(self.sampling_config),
                ):
                    self.context.token_nums += 1
                    yield resp
                ed = time.perf_counter_ns()
                logger.debug(
                    f"Generate stream latency: {(ed - st) / 1e6} ms. pid={self.pid}, tid={self.tid}"
                )
        except BaseException as e:
            logger.error(f"Generate error in {self.context.engine_url} error: {e}")
            raise e
