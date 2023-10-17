import aiohttp
from dataclasses import asdict

from parrot.engine.config import EngineConfig

from .responses import (
    VMHeartbeatResponse,
    RegisterVMResponse,
    SubmitCallResponse,
    PlaceholderFetchResponse,
    EngineHeartbeatResponse,
    FreeContextResponse,
    RegisterEngineResponse,
)
from .common import (
    send_http_request,
    async_send_http_request,
    logger,
)


# ---------- Program Layer to OS Layer APIs ----------


def vm_heartbeat(
    http_addr: str,
    pid: int,
) -> VMHeartbeatResponse:
    try:
        return send_http_request(
            VMHeartbeatResponse,
            http_addr,
            "/vm_heartbeat",
            retry_times=3,
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
            retry_times=3,
        )
    except BaseException as e:
        logger.error(f"Register VM error in {http_addr}. Error: {e}")
        raise e


def submit_call(http_addr: str, pid: int, call: "SemanticCall") -> SubmitCallResponse:
    try:
        return send_http_request(
            SubmitCallResponse,
            http_addr,
            "/submit_call",
            retry_times=3,
            pid=pid,
            call=call.pickle(),
        )
    except BaseException as e:
        logger.error(f"Execute func (pid: {pid}) error in {http_addr}. Error: {e}")
        raise e


def placeholder_fetch(
    http_addr: str, pid: int, placeholder_id: int
) -> PlaceholderFetchResponse:
    try:
        return send_http_request(
            PlaceholderFetchResponse,
            http_addr,
            "/placeholder_fetch",
            retry_times=3,
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
                retry_times=3,
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


# ---------- Engine Layer to OS Layer APIs ----------


def register_engine(
    http_addr: str,
    engine_name: str,
    engine_config: EngineConfig,
) -> RegisterEngineResponse:
    try:
        return send_http_request(
            RegisterEngineResponse,
            http_addr,
            "/register_engine",
            retry_times=3,
            engine_name=engine_name,
            engine_config=asdict(engine_config),
        )
    except BaseException as e:
        logger.error(f"Register engine {engine_name} error in {http_addr}. Error: {e}")
        raise e


def engine_heartbeat(
    http_addr: str,
    engine_name: str,
    num_running_jobs: int,
    num_cached_tokens: int,
    cache_mem: float,
    model_mem: float,
) -> EngineHeartbeatResponse:
    try:
        return send_http_request(
            EngineHeartbeatResponse,
            http_addr,
            "/engine_heartbeat",
            retry_times=3,
            engine_name=engine_name,
            num_running_jobs=num_running_jobs,
            num_cached_tokens=num_cached_tokens,
            cache_mem=cache_mem,
            model_mem=model_mem,
        )
    except BaseException as e:
        logger.error(f"Check engine heartbeat error. Engine: {engine_name}, Error: {e}")
        raise e
