# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import asyncio
import multiprocessing as mp
import parrot as P
from parrot.utils import cprofile
import numpy as np


def proc1(barrier: mp.Barrier):
    vm = P.VirtualMachine(
        os_http_addr="http://localhost:9000",
        mode="debug",
    )

    test_func = vm.import_function(
        "chain_sum_test",
        "benchmark.workloads.test_examples.chain_summarization",
    )

    input_workload = "Test " * 100

    chunk_num = 20

    async def main_async():
        outputs = [P.variable(name=f"output_{i}") for i in range(chunk_num)]
        coroutines = []
        for i in range(chunk_num):
            if i == 0:
                coro = test_func.ainvoke(
                    previous_document=input_workload, refined_document=outputs[i]
                )
            else:
                coro = test_func.ainvoke(
                    previous_document=outputs[i - 1], refined_document=outputs[i]
                )
            coroutines.append(coro)
        await asyncio.gather(*coroutines)
        outputs[-1].get()

    barrier.wait()

    vm.run(main_async, timeit=True)


def proc2(barrier: mp.Barrier, request_rate: float):
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    test_func = vm.import_function(
        "func_1i_1o_genlen_100", "benchmark.workloads.test_examples.normal_functions"
    )

    requests_num = 100

    outputs = []

    with vm.running_scope():
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
    p1 = mp.Process(
        target=proc1,
        args=(barrier,),
    )
    p2 = mp.Process(
        target=proc2,
        args=(
            barrier,
            request_rate,
        ),
    )

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == "__main__":
    # main(1.0)
    main(1.0)
