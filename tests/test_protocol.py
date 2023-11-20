import json
import time
import asyncio

import parrot as P

from parrot.engine.runtime_info import EngineRuntimeInfo
from parrot.engine.config import EngineConfig
from parrot.os.engine import ExecutionEngine
from parrot.os.memory.context import Context
from parrot.constants import NONE_THREAD_ID

from parrot.protocol.layer_apis import (
    vm_heartbeat,
    register_vm,
    submit_call,
    placeholder_fetch,
    free_context,
    ping_engine,
    engine_heartbeat,
    register_engine,
)
from parrot.protocol.primitive_request import Fill, Generate
from parrot.protocol.sampling_config import SamplingConfig

from parrot.testing.fake_os_server import TESTING_SERVER_URL as OS_URL
from parrot.testing.fake_engine_server import TESTING_SERVER_URL as ENGINE_URL
from parrot.testing.fake_engine_server import TESTING_SERVER_HOST, TESTING_SERVER_PORT
from parrot.testing.localhost_server_daemon import fake_os_server, fake_engine_server
from parrot.testing.get_configs import get_engine_config_path


def test_vm_heartbeat():
    with fake_os_server():
        # st = time.perf_counter_ns()
        resp = vm_heartbeat(http_addr=OS_URL, pid=0)
        # ed = time.perf_counter_ns()
        # print("Heartbeat Time Used: ", (ed - st) / 1e9)

        assert resp.mem_used == 0.0
        assert resp.num_threads == 0


def test_register_vm():
    with fake_os_server():
        resp = register_vm(
            http_addr=OS_URL,
        )

        assert (
            resp.pid == 0
        )  # It's related to the allocating policy of the fake OS server


def test_submit_call():
    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    call = test("a", b="b")

    with fake_os_server():
        resp = submit_call(
            http_addr=OS_URL,
            pid=0,
            call=call,
        )


def test_placeholder_fetch():
    with fake_os_server():
        resp = placeholder_fetch(
            http_addr=OS_URL,
            pid=0,
            placeholder_id=0,
        )

        assert resp.content == "placeholder_xxx"


def test_free_context():
    with fake_engine_server():
        resp = free_context(
            http_addr=ENGINE_URL,
            context_id=0,
        )

        assert resp.context_len == 0


def test_ping_engine():
    with fake_engine_server():
        assert ping_engine(http_addr=ENGINE_URL)


def test_engine_heartbeat():
    with fake_os_server():
        resp = engine_heartbeat(
            http_addr=OS_URL,
            engine_id=0,
            engine_name="test",
            runtime_info=EngineRuntimeInfo(),
        )


def _get_opt_125m_engine_config():
    engine_config_path = get_engine_config_path("opt-125m.json")
    with open(engine_config_path, "r") as f:
        engine_config = json.load(f)

    assert EngineConfig.verify_config(engine_config)
    engine_config.pop("instance")
    engine_config.pop("scheduler")
    engine_config.pop("os")
    engine_config = EngineConfig(**engine_config)
    engine_config.host = TESTING_SERVER_HOST
    engine_config.port = TESTING_SERVER_PORT
    return engine_config


def test_register_engine():
    engine_config = _get_opt_125m_engine_config()

    with fake_os_server():
        resp = register_engine(
            http_addr=OS_URL,
            engine_config=engine_config,
        )

        assert (
            resp.engine_id == 0
        )  # It's related to the allocating policy of the fake OS server


def test_fill():
    engine_config = _get_opt_125m_engine_config()
    engine = ExecutionEngine(engine_id=0, config=engine_config)
    context = Context(context_id=0, engine=engine)

    async def main():
        primitve = Fill(
            pid=0,
            tid=NONE_THREAD_ID,
            context=context,
            token_ids=[1, 2, 3],
        )
        st = time.perf_counter_ns()
        resp = primitve.post()
        ed = time.perf_counter_ns()
        print("Fill Time Used: ", (ed - st) / 1e9)
        assert resp.filled_len == 3

        primitve.tid = 0  # Now we have a tid
        resp = await primitve.apost()
        assert resp.filled_len == 3

    with fake_engine_server():
        asyncio.run(main())


def test_generate():
    engine_config = _get_opt_125m_engine_config()
    engine = ExecutionEngine(engine_id=0, config=engine_config)
    context = Context(context_id=0, engine=engine)

    async def main():
        primitive = Generate(
            pid=0,
            tid=0,
            context=context,
            sampling_config=SamplingConfig(),
        )

        # Generate
        st = time.perf_counter_ns()
        resp = await primitive.apost()
        ed = time.perf_counter_ns()
        print("Generate Time Used: ", (ed - st) / 1e9, "(s)")

        # Generate Stream
        counter = 0
        times = []

        st = time.perf_counter_ns()
        async for token_id in primitive.astream():
            counter += 1
            # assert counter == token_id
            # print(token_id)
            cur_time = time.perf_counter_ns()
            times.append((cur_time - st) / 1e9)
            st = cur_time

        print("Generation Time Points: ", times)

    with fake_engine_server():
        asyncio.run(main())


if __name__ == "__main__":
    test_vm_heartbeat()
    test_register_vm()
    test_submit_call()
    test_placeholder_fetch()
    test_free_context()
    test_ping_engine()
    test_engine_heartbeat()
    test_register_engine()
    test_fill()
    test_generate()
