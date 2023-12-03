from parrot.testing.latency_simulator import get_latency


def test_simulate_latency():
    for _ in range(10):
        latency = get_latency()
        print(latency)


if __name__ == "__main__":
    test_simulate_latency()
