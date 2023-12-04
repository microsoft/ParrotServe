# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import asyncio
import multiprocessing as mp
import parrot as P
from parrot.utils import cprofile
import numpy as np


def proc1(barrier: mp.Barrier):
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

    barrier.wait()

    vm.run(main, timeit=True)


def proc2(barrier: mp.Barrier, request_rate: float):
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    test_func = vm.import_function(
        "func_1i_1o_genlen_100", "benchmark.workloads.test_examples.normal_functions"
    )

    requests_num = 100

    outputs = []

    barrier.wait()

    for _ in range(requests_num):
        output = test_func("Test")
        outputs.append(output)
        interval = np.random.exponential(1.0 / request_rate)
        time.sleep(interval)

    for i in range(requests_num):
        outputs[i].get()


def main(request_rate: int):
    barrier = mp.Barrier(2)
    proc1 = mp.Process(
        target=proc1,
        args=(barrier,),
    )
    proc2 = mp.Process(
        target=proc2,
        args=(
            barrier,
            request_rate,
        ),
    )

    proc1.start()
    proc2.start()

    proc1.join()
    proc2.join()


if __name__ == "__main__":
    main()
