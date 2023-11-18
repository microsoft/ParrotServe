# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import parrot as P

import cProfile, pstats, io

vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

test_func = vm.import_function("chain_sum_test", "bench.chain_summarization")

input_workload = "Test " * 100

chunk_num = 20


def main():
    # NOTE(chaofan): We only get the final result, let the intermediate results be
    # flown in the system.
    next_input = input_workload
    for _ in range(chunk_num):
        next_input = test_func(next_input)
    next_input.get()


def baseline():
    # NOTE(chaofan): For baseline, we call `get` for every summarization, which means
    # they are executed sequentially.
    next_input = input_workload
    for _ in range(chunk_num):
        next_input = test_func(next_input)
        next_input.get()


def test_baseline():
    print("baseline:")
    vm.run(baseline, timeit=True)
    time.sleep(3)


def test_main():
    print("main:")
    vm.run(main, timeit=True)
    time.sleep(3)


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    # print(test_func.body)

    test_baseline()
    # test_main()

    # for _ in range(5):
    #     test_baseline()
    #     test_main()

    # latency = vm.profile(main)
    # print(latency)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(2)
    ps.print_stats()
    print(s.getvalue())
