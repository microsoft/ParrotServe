# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import asyncio
import parrot as P
from parrot.utils import cprofile
import numpy as np


def proc1():
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    test_func = vm.import_function(
        "chain_sum_test",
        "benchmark.workloads.test_examples.chain_summarization",
        mode="debug",
    )

    input_workload = "Test " * 100

    chunk_num = 20

    def main():
        next_input = input_workload
        for _ in range(chunk_num):
            next_input = test_func(next_input)
        next_input.get()

    def baseline():
        next_input = input_workload
        for _ in range(chunk_num):
            next_input = test_func(next_input)
            # with cprofile("get"):
            next_input.get()

    vm.run(main, timeit=True)


def proc2(request_rate: float):
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    test_func = vm.import_function(
        "func_1i_1o_genlen_100", "benchmark.workloads.test_examples.normal_functions"
    )

    requests_num = 100

    outputs = []

    for _ in range(requests_num):
        output = test_func("Test")
        outputs.append(output)
        interval = np.random.exponential(1.0 / request_rate)
        time.sleep(interval)

    for i in range(requests_num):
        outputs[i].get()


if __name__ == "__main__":
    # print(test_func.body)

    # with cprofile("e2e"):
    #     vm.run(single_call, timeit=True)
    # vm.run(single_call, timeit=True)

    # test_baseline()
    test_main()
    # test_main_async()

    # for _ in range(10):
    # test_baseline()
    # test_main()
    #   test_main_async()

    # latency = vm.profile(main)
    # print(latency)
