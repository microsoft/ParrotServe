from typing import List
from dataclasses import asdict

from parrot.backend.config import EngineConfig

from .responses import (
    VMHeartbeatResponse,
    RegisterVMResponse,
    ThreadStartResponse,
    ThreadEndResponse,
    EngineHeartbeatResponse,
    FreeContextResponse,
    RegisterEngineResponse,
)
from .thread_metadata import ThreadMetadata
from .common import (
    send_http_request,
    logger,
)


# ---------- VM Layer to OS Layer APIs ----------


def vm_heartbeat(
    http_addr: str,
    pid: int,
):
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


def register_vm(http_addr: str):
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


def thread_start(http_addr: str, pid: int, tid: int, metadata: ThreadMetadata):
    try:
        return send_http_request(
            ThreadStartResponse,
            http_addr,
            "/thread_start",
            retry_times=3,
            pid=pid,
            tid=tid,
            metadata=metadata,
        )
    except BaseException as e:
        logger.error(f"Start thread (pid: {pid}) error in {http_addr}. Error: {e}")
        raise e


def thread_end(http_addr: str, pid: int, tid: int):
    try:
        return send_http_request(
            ThreadEndResponse,
            http_addr,
            "/thread_end",
            retry_times=3,
            pid=pid,
            tid=tid,
        )
    except BaseException as e:
        logger.error(f"End thread (pid: {pid}) error in {http_addr}. Error: {e}")
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
):
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
    cached_tokens_size: float,
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
            cached_tokens_size=cached_tokens_size,
        )
    except BaseException as e:
        logger.error(f"Check engine heartbeat error. Engine: {engine_name}, Error: {e}")
        raise e
