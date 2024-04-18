import json
import time
import asyncio

from parrot.protocol.internal.runtime_info import EngineRuntimeInfo
from parrot.engine.config import EngineConfig
from parrot.serve.backend_repr import ExecutionEngine, LanguageModel
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.backend_repr.context import Context
from parrot.constants import NONE_THREAD_ID

from parrot.protocol.public.apis import (
    register_session,
    get_session_info,
    remove_session,
    submit_semantic_call,
    register_semantic_variable,
    set_semantic_variable,
    get_semantic_variable,
    get_semantic_variable_list,
)
from parrot.protocol.internal.layer_apis import (
    free_context,
    ping_engine,
    engine_heartbeat,
    register_engine,
)
from parrot.protocol.internal.primitive_request import Fill, Generate
from parrot.sampling_config import SamplingConfig

from parrot.testing.fake_core_server import TESTING_SERVER_URL as CORE_URL
from parrot.testing.fake_engine_server import TESTING_SERVER_URL as ENGINE_URL
from parrot.testing.fake_engine_server import TESTING_SERVER_HOST, TESTING_SERVER_PORT
from parrot.testing.localhost_server_daemon import fake_core_server, fake_engine_server
from parrot.testing.get_configs import get_sample_engine_config_path


def test_register_session():
    with fake_core_server():
        resp = register_session(http_addr=CORE_URL, api_key="1")
        assert resp.session_id == 0


def test_remove_session():
    with fake_core_server():
        resp = register_session(http_addr=CORE_URL, api_key="1")
        resp1 = remove_session(
            http_addr=CORE_URL, session_id=resp.session_id, session_auth="1"
        )


def test_submit_semantic_call():
    payload = {
        "template": "This is a test {{a}} function. {{b}}",
        "placeholders": [
            {
                "name": "a",
                "is_output": False,
                "var_id": "xxx",
            },
            {
                "name": "b",
                "is_output": True,
                "sampling_config": {
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
            },
        ],
        "models": ["model1", "model2"],
        "model_type": "token_id",
        "remove_pure_fill": True,
    }

    with fake_core_server():
        resp = submit_semantic_call(
            http_addr=CORE_URL,
            session_id=0,
            session_auth="1",
            payload=payload,
        )

        assert resp.request_id == 0


def test_register_semantic_variable():
    with fake_core_server():
        resp = register_semantic_variable(
            http_addr=CORE_URL,
            session_id=0,
            session_auth="1",
            var_name="test",
        )

        print(resp.var_id)


def test_set_semantic_variable():
    with fake_core_server():
        resp = register_semantic_variable(
            http_addr=CORE_URL,
            session_id=0,
            session_auth="1",
            var_name="test",
        )

        print(resp.var_id)

        resp1 = set_semantic_variable(
            http_addr=CORE_URL,
            session_id=0,
            session_auth="1",
            var_id=resp.var_id,
            content="test_value",
        )


def test_get_semantic_variable():
    with fake_core_server():
        resp = register_semantic_variable(
            http_addr=CORE_URL,
            session_id=0,
            session_auth="1",
            var_name="test",
        )

        print(resp.var_id)
        content = "test_value"

        resp1 = set_semantic_variable(
            http_addr=CORE_URL,
            session_id=0,
            session_auth="1",
            var_id=resp.var_id,
            content=content,
        )

        resp2 = get_semantic_variable(
            http_addr=CORE_URL,
            session_id=0,
            session_auth="1",
            var_id=resp.var_id,
            criteria="latency",
        )

        assert resp2.content == content


def test_free_context():
    with fake_engine_server():
        resp = free_context(
            http_addr=ENGINE_URL,
            context_id=0,
        )

        assert resp.context_len == 0


def test_ping_engine():
    with fake_engine_server():
        resp = ping_engine(http_addr=ENGINE_URL)
        assert resp.pong


def test_engine_heartbeat():
    with fake_core_server():
        resp = engine_heartbeat(
            http_addr=CORE_URL,
            engine_id=0,
            engine_name="test",
            runtime_info=EngineRuntimeInfo(),
        )


def _get_opt_125m_engine_config():
    engine_config_path = get_sample_engine_config_path("opt-125m.json")
    with open(engine_config_path, "r") as f:
        engine_config = json.load(f)

    assert EngineConfig.verify_config(engine_config)
    engine_config = EngineConfig.from_dict(engine_config)
    engine_config.host = TESTING_SERVER_HOST
    engine_config.port = TESTING_SERVER_PORT
    return engine_config


def test_register_engine():
    engine_config = _get_opt_125m_engine_config()

    with fake_core_server():
        resp = register_engine(
            http_addr=CORE_URL,
            engine_config=engine_config,
        )

        assert (
            resp.engine_id == 0
        )  # It's related to the allocating policy of the fake core server


def test_fill():
    engine_config = _get_opt_125m_engine_config()
    engine = ExecutionEngine.from_engine_config(engine_id=0, config=engine_config)

    async def main():
        primitve = Fill(
            session_id=0,
            task_id=0,
            context_id=0,
            parent_context_id=-1,
            end_flag=False,
            token_ids=[1, 2, 3],
        )
        st = time.perf_counter_ns()
        resp = primitve.post(engine.http_address)
        ed = time.perf_counter_ns()
        print("Fill Time Used: ", (ed - st) / 1e9)
        assert resp.filled_len == 3
        resp = await primitve.apost(engine.http_address)
        assert resp.filled_len == 3

    with fake_engine_server():
        asyncio.run(main())


def test_generate():
    engine_config = _get_opt_125m_engine_config()
    model = LanguageModel.from_engine_config(engine_config)
    engine = ExecutionEngine(
        engine_id=0,
        config=engine_config,
        model=model,
    )

    async def main():
        primitive = Generate(
            session_id=0,
            task_id=0,
            context_id=0,
            parent_context_id=-1,
            end_flag=False,
            sampling_config=SamplingConfig(),
        )

        # Generate
        st = time.perf_counter_ns()
        resp = await primitive.apost(engine.http_address)
        ed = time.perf_counter_ns()
        print(
            "Generate Time Used: ",
            (ed - st) / 1e9,
            f"(s), generated tokens: {len(resp.generated_ids)}",
        )

        # Generate Stream
        counter = 0
        times = []

        st = time.perf_counter_ns()
        async for token_id in primitive.astream(engine.http_address):
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
    # test_register_session()
    # test_remove_session()
    test_submit_semantic_call()
    # test_register_semantic_variable()
    # test_set_semantic_variable()
    # test_get_semantic_variable()
    # test_free_context()
    # test_fill()
    # test_generate()
    pass
