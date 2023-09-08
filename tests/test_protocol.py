"""This test requires a running backend server.
Use `python3 -m parrot.testing.fake_server` to start a fake server.
"""


from parrot.protocol import check_heartbeat, prefix_init, fill, generate, SamplingParams
from aiohttp import ClientSession
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

    assert resp.model_ready == True
    assert resp.cached_tokens == 0
    assert resp.running_jobs == 0


def test_prefix_init():
    resp = prefix_init(
        http_addr=addr,
        context_id=0,
        token_ids=[1, 2, 3],
    )

    assert resp.filled_tokens_num == 3


def test_fill():
    async def main():
        async with ClientSession() as session:
            st = time.perf_counter_ns()
            resp = await fill(
                client_session=session,
                http_addr=addr,
                session_id=0,
                context_id=0,
                parent_context_id=-1,
                token_ids=[1, 2, 3],
            )
            ed = time.perf_counter_ns()
            print("Fill Time Used: ", (ed - st) / 1e9)

            assert resp.filled_tokens_num == 3

    asyncio.run(main())


def test_generate():
    times = []

    async def main():
        counter = 0
        st = time.perf_counter_ns()
        async with ClientSession() as session:
            async for token_id in generate(
                client_session=session,
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


if __name__ == "__main__":
    test_check_heartbeat()
    test_prefix_init()
    test_fill()
    test_generate()
