# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

from numpy import mean
import time
import parrot as P
from multiprocessing import Barrier
from parrot.testing.multiproc_manager import MultiProcessManager
from parrot.utils import cprofile

input_workload = "Test " * 100

chunk_num = 20
clients_num = 8


def main(test_func):
    # NOTE(chaofan): We only get the final result, let the intermediate results be
    # flown in the system.
    next_input = input_workload
    for _ in range(chunk_num):
        next_input = test_func(next_input)
    next_input.get()


def baseline(test_func):
    # NOTE(chaofan): For baseline, we call `get` for every summarization, which means
    # they are executed sequentially.
    next_input = input_workload
    for _ in range(chunk_num):
        next_input = test_func(next_input)
        # with cprofile("get"):
        next_input.get()


def process(barrier: Barrier, is_baseline: bool = True):
    vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

    test_func = vm.import_function(
        "chain_sum_test", "benchmark.workloads.test_examples.chain_summarization"
    )

    proc_func = baseline if is_baseline else main

    barrier.wait()

    latency = vm.run(
        proc_func,
        timeit=True,
        args=[test_func],
    )

    return latency


def test_baseline():
    print("baseline:")

    manager = MultiProcessManager()
    barrier = Barrier(clients_num)

    for _ in range(clients_num):
        manager.add_proc(process, (barrier, True))

    manager.run_all()
    print(manager.data)
    time.sleep(3)


def test_main():
    print("main:")

    manager = MultiProcessManager()
    barrier = Barrier(clients_num)

    for _ in range(clients_num):
        manager.add_proc(process, (barrier, False))

    manager.run_all()
    print(manager.data)
    print(f"Avg. JCT {mean(list(manager.data.values())):.2f} (s)")
    time.sleep(3)


if __name__ == "__main__":
    # test_baseline()
    test_main()
