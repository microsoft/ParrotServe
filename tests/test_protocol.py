"""This test requires a running backend server.
Use `python3 -m parrot.testing.fake_server` to start a fake server.
"""


from parrot.protocol import (
    check_heartbeat,
    fill,
    afill,
    agenerate,
    free_context,
    SamplingParams,
)
import time
import asyncio


addr = "http://localhost:8888"


def test_check_heartbeat():
    st = time.perf_counter_ns()
    resp = check_heartbeat(
        engine_name="test",
        http_addr=addr,
    )
    ed = time.perf_counter_ns()
    print("Heartbeat Time Used: ", (ed - st) / 1e9)

    assert resp.num_cached_tokens == 0
    assert resp.cached_tokens_size == 0
    assert resp.num_running_jobs == 0


def test_prefix_init():
    resp = fill(
        http_addr=addr,
        context_id=0,
        parent_context_id=-1,
        token_ids=[1, 2, 3],
    )

    assert resp.num_filled_tokens == 3


def test_fill():
    async def main():
        st = time.perf_counter_ns()
        resp = await afill(
            http_addr=addr,
            session_id=0,
            context_id=0,
            parent_context_id=-1,
            token_ids=[1, 2, 3],
        )
        ed = time.perf_counter_ns()
        print("Fill Time Used: ", (ed - st) / 1e9)

        assert resp.num_filled_tokens == 3

    asyncio.run(main())


def test_generate():
    times = []

    async def main():
        counter = 0
        st = time.perf_counter_ns()
        async for token_id in agenerate(
            http_addr=addr,
            session_id=0,
            context_id=0,
            parent_context_id=-1,
            sampling_params=SamplingParams(),
        ):
            counter += 1
            # assert counter == token_id
            # print(token_id)
            cur_time = time.perf_counter_ns()
            times.append((cur_time - st) / 1e9)
            st = cur_time

    asyncio.run(main())
    print("Generation Time Points: ", times)


def test_free_context():
    resp = free_context(
        http_addr=addr,
        context_id=233,
    )

    assert resp.num_freed_tokens == 0


if __name__ == "__main__":
    test_check_heartbeat()
    test_prefix_init()
    test_fill()
    test_generate()
    test_free_context()
