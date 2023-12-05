# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import aiohttp
from dataclasses import asdict

from .responses import (
    VMHeartbeatResponse,
    RegisterVMResponse,
    SubmitCallResponse,
    PlaceholderSetResponse,
    PlaceholderFetchResponse,
    EngineHeartbeatResponse,
    FreeContextResponse,
    RegisterEngineResponse,
    PingResponse,
)
from .common import (
    send_http_request,
    async_send_http_request,
    logger,
)
from .runtime_info import EngineRuntimeInfo


# ---------- Program Layer to OS Layer APIs ----------


def vm_heartbeat(
    http_addr: str,
    pid: int,
) -> VMHeartbeatResponse:
    try:
        return send_http_request(
            response_cls=VMHeartbeatResponse,
            http_addr=http_addr,
            api_url="/vm_heartbeat",
            retry_times=3,
            # timeout=3,
            pid=pid,
        )
    except BaseException as e:
        logger.error(
            f"Check vm (pid: {pid}) heartbeat error in {http_addr}. Error: {e}"
        )
        raise e


def register_vm(http_addr: str) -> RegisterVMResponse:
    try:
        return send_http_request(
            RegisterVMResponse,
            http_addr,
            "/register_vm",
            retry_times=1,
        )
    except BaseException as e:
        logger.error(f"Register VM error in {http_addr}. Error: {e}")
        raise e


def submit_call(
    http_addr: str, pid: int, call: "BasicCall", is_native: bool
) -> SubmitCallResponse:
    try:
        return send_http_request(
            SubmitCallResponse,
            http_addr,
            "/submit_call",
            retry_times=1,
            pid=pid,
            call=call.pickle(),
            is_native=is_native,
        )
    except BaseException as e:
        logger.error(f"Execute func (pid: {pid}) error in {http_addr}. Error: {e}")
        raise e


async def asubmit_call(
    http_addr: str, pid: int, call: "BasicCall", is_native: bool
) -> SubmitCallResponse:
    try:
        async with aiohttp.ClientSession() as client_session:
            return await async_send_http_request(
                client_session,
                SubmitCallResponse,
                http_addr,
                "/submit_call",
                retry_times=1,
                pid=pid,
                call=call.pickle(),
                is_native=is_native,
            )
    except BaseException as e:
        logger.error(f"Execute func (pid: {pid}) error in {http_addr}. Error: {e}")
        raise e


def placeholder_set(http_addr: str, pid: int, placeholder_id: int, content: str):
    try:
        send_http_request(
            PlaceholderSetResponse,
            http_addr,
            "/placeholder_set",
            retry_times=1,
            pid=pid,
            placeholder_id=placeholder_id,
            content=content,
        )
    except BaseException as e:
        logger.error(f"Placeholder set (pid: {pid}) error in {http_addr}. Error: {e}")
        raise e


def placeholder_fetch(
    http_addr: str, pid: int, placeholder_id: int
) -> PlaceholderFetchResponse:
    try:
        return send_http_request(
            PlaceholderFetchResponse,
            http_addr,
            "/placeholder_fetch",
            retry_times=1,
            pid=pid,
            placeholder_id=placeholder_id,
        )
    except BaseException as e:
        logger.error(f"Placeholder fetch (pid: {pid}) error in {http_addr}. Error: {e}")
        raise e


async def aplaceholder_fetch(
    http_addr: str, pid: int, placeholder_id: int
) -> PlaceholderFetchResponse:
    try:
        async with aiohttp.ClientSession() as client_session:
            return await async_send_http_request(
                client_session,
                PlaceholderFetchResponse,
                http_addr,
                "/placeholder_fetch",
                retry_times=1,
                pid=pid,
                placeholder_id=placeholder_id,
            )
    except BaseException as e:
        logger.error(f"Placeholder fetch (pid: {pid}) error in {http_addr}. Error: {e}")
        raise e


# ---------- OS Layer to Engine Layer APIs ----------


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
        logger.error(f"Free context error in {http_addr}. Error: {e}")
        raise e


def ping_engine(http_addr: str) -> PingResponse:
    try:
        return send_http_request(
            PingResponse,
            http_addr,
            "/ping",
            retry_times=5,
        )
    except BaseException as e:
        print(e.args)
        return PingResponse(pong=False)


# ---------- Engine Layer to OS Layer APIs ----------


def register_engine(
    http_addr: str,
    engine_config: "EngineConfig",
) -> RegisterEngineResponse:
    try:
        return send_http_request(
            RegisterEngineResponse,
            http_addr,
            "/register_engine",
            retry_times=1,
            engine_config=asdict(engine_config),
        )
    except BaseException as e:
        logger.error(
            f"Register engine {engine_config.engine_name} error in {http_addr}. Error: {e}"
        )
        raise e


def engine_heartbeat(
    http_addr: str,
    engine_id: int,
    engine_name: str,
    runtime_info: EngineRuntimeInfo,
) -> EngineHeartbeatResponse:
    try:
        return send_http_request(
            response_cls=EngineHeartbeatResponse,
            http_addr=http_addr,
            api_url="/engine_heartbeat",
            retry_times=3,
            engine_id=engine_id,
            engine_name=engine_name,
            runtime_info=asdict(runtime_info),
        )
    except BaseException as e:
        logger.error(
            f"Check engine heartbeat error. Engine: {engine_name} (id={engine_id}), Error: {e}"
        )
        raise e
