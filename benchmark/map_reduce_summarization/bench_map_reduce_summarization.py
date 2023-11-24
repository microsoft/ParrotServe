# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

import time
import parrot as P
from parrot.utils import cprofile
from parrot.testing.localhost_server_daemon import fake_os_server

vm = P.VirtualMachine(os_http_addr="http://localhost:9000")
map_func_test = vm.import_function("map_sum_test", "bench.map_reduce_summarization")
reduce_func_test = vm.import_function("reduce_sum_test", "bench.map_reduce_summarization")

chunk_num = len(reduce_func_test.inputs)

map_document_chunk = "Test " * 1024  # len=1024 for each chunk


def main():
    # NOTE(chaofan): We annotate map stage a large job upperbound, 
    # and reduce stage a small job upperbound.
    map_func_test.outputs[0].dispatch_annotation.requests_num_upperbound = chunk_num
    reduce_func_test.outputs[0].dispatch_annotation.requests_num_upperbound = 2

    chunk_sums = []
    for _ in range(chunk_num):
        chunk_sums.append(map_func_test(map_document_chunk)) 
    final_output = reduce_func_test(*chunk_sums)
    final_output.get()


def baseline():
    # NOTE(chaofan): For baseline, both map and reduce stage are small upperbound.
    map_func_test.outputs[0].dispatch_annotation.requests_num_upperbound = 2
    reduce_func_test.outputs[0].dispatch_annotation.requests_num_upperbound = 2

    chunk_sums = []
    map_input = P.future()
    for _ in range(chunk_num):
        chunk_sums.append(map_func_test(map_input))
    time.sleep(3) # take 3 second to load the docs from web, suppose
    map_input.set(map_document_chunk)

    final_output = reduce_func_test(*chunk_sums)
    final_output.get()


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
    test_baseline()
    test_main()
