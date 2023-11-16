import time

from parrot.protocol.layer_apis import ping_engine
from parrot.testing.localhost_server_daemon import fake_engine_server
from parrot.constants import DEFAULT_ENGINE_URL


def test_latency():
    warmups = 10
    test_iters = 100

    with fake_engine_server():
        for _ in range(warmups):
            ping_engine(
                http_addr=DEFAULT_ENGINE_URL,
            )

        st = time.perf_counter_ns()
        for i in range(test_iters):
            ping_engine(
                http_addr=DEFAULT_ENGINE_URL,
            )
            print(f"Sent ping request: {i}")
        ed = time.perf_counter_ns()

        print(f"Average latency: {(ed - st) / test_iters / 1e6} ms")


if __name__ == "__main__":
    test_latency()
