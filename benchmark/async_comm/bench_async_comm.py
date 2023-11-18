# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import parrot as P
from parrot.utils import cprofile

vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

test_func = vm.import_function("chain_sum_test", "bench.chain_summarization")

input_workload = "Test " * 100

chunk_num = 20


def single_call():
    holder = test_func(input_workload)
    with cprofile("get"):
        holder.get()


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
        # with cprofile("get"):
        next_input.get()


def test_baseline():
    print("baseline:")
    # with cprofile("baseline"):
    vm.run(baseline, timeit=True)
    time.sleep(3)


def test_main():
    print("main:")
    # with cprofile("main"):
    vm.run(main, timeit=True)
    time.sleep(3)


if __name__ == "__main__":
    # print(test_func.body)

    # vm.run(single_call, timeit=True)

    test_baseline()
    # test_main()

    # for _ in range(5):
    #     test_baseline()
    #     test_main()

    # latency = vm.profile(main)
    # print(latency)
