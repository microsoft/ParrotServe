# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Start specific daemon servers in localhost for testing.

The server daemon will run in a separate process created by the lib Python multiprocessing.
"""


from typing import Dict
import contextlib

import torch

import uvicorn
import time

from parrot.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_CORE_SERVER_PORT,
    DEFAULT_ENGINE_SERVER_PORT,
)

from parrot.serve.http_server import start_server as start_core_server
from parrot.engine.http_server import start_server as start_engine_server

from .get_configs import get_sample_engine_config_path, get_sample_core_config_path
from .fake_engine_server import app as FakeEngineApp
from .fake_core_server import app as FakeCoreApp

# RuntimeError: Cannot re-initialize CUDA in forked subprocess.
# To use CUDA with multiprocessing, you must use the 'spawn' start method
# torch.multiprocessing.set_start_method("spawn")

# Issue: https://github.com/pytorch/pytorch/issues/3492

from torch import multiprocessing

ctx = multiprocessing.get_context("spawn")
TorchProcess = ctx.Process


from multiprocessing import Process as StdProcess


# NOTE(chaofan): Do not use closure here, since the torch "spawn" method
# need to pickle the function.
def _launch_fake_core():
    uvicorn.run(
        FakeCoreApp,
        host=DEFAULT_SERVER_HOST,
        port=DEFAULT_CORE_SERVER_PORT,
        log_level="info",
    )


@contextlib.contextmanager
def fake_core_server():
    p = StdProcess(target=_launch_fake_core, daemon=True)
    p.start()
    time.sleep(0.1)

    yield

    p.terminate()
    time.sleep(0.1)


def _launch_fake_engine():
    uvicorn.run(
        FakeEngineApp,
        host=DEFAULT_SERVER_HOST,
        port=DEFAULT_ENGINE_SERVER_PORT,
        log_level="info",
    )


@contextlib.contextmanager
def fake_engine_server():
    p = StdProcess(target=_launch_fake_engine, daemon=True)
    p.start()
    time.sleep(0.1)

    yield

    p.terminate()
    time.sleep(0.1)


def _launch_core():
    core_config_path = get_sample_core_config_path("localhost_serve_core.json")
    release_mode = False

    start_core_server(core_config_path=core_config_path, release_mode=release_mode)


@contextlib.contextmanager
def core_server():
    p = StdProcess(target=_launch_core, daemon=True)
    p.start()
    time.sleep(0.1)

    yield

    p.terminate()
    time.sleep(0.1)


def _launch_engine(engine_config_name: str, connect_to_core: bool, override_args: Dict):
    engine_config_path = get_sample_engine_config_path(engine_config_name)
    start_engine_server(
        engine_config_path=engine_config_path,
        connect_to_core=connect_to_core,
        override_args=override_args,
    )


@contextlib.contextmanager
def engine_server(
    engine_config_name: str,
    wait_ready_time: float = 0.1,
    connect_to_core: bool = False,
    **args,
):
    override_args = args
    p = TorchProcess(
        target=_launch_engine,
        args=(
            engine_config_name,
            connect_to_core,
            override_args,
        ),
        daemon=True,
    )
    p.start()
    time.sleep(wait_ready_time)

    yield

    p.terminate()
    time.sleep(0.1)


@contextlib.contextmanager
def system_opt():
    with core_server():
        with engine_server(
            engine_config_name="opt-125m.json",
            wait_ready_time=3.0,
            connect_to_core=True,
        ):
            yield


@contextlib.contextmanager
def system_vicuna():
    with core_server():
        with engine_server(
            engine_config_name="vicuna-7b-v1.3.json",
            wait_ready_time=5.0,
            connect_to_core=True,
        ):
            yield


@contextlib.contextmanager
def system_vicuna_vllm():
    with core_server():
        with engine_server(
            engine_config_name="vicuna-7b-v1.3-vllm.json",
            wait_ready_time=5.0,
            connect_to_core=True,
        ):
            yield


@contextlib.contextmanager
def system_openai():
    with core_server():
        with engine_server(
            engine_config_name="azure-openai-gpt-3.5-turbo.json",
            wait_ready_time=3.0,
            connect_to_core=True,
        ):
            yield
