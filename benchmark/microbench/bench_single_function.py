# Copyright (c) 2023 by Microsoft Corporation.
# Author: Chaofan Lin (v-chaofanlin@microsoft.com)

# 8.455 (s): Native vicuna-13B

import parrot as P

vm = P.VirtualMachine(os_http_addr="http://localhost:9000")

test_func = vm.import_function("func_3i_2o_genlen_100", "bench.normal_functions")


def main():
    input_workload = "This is a test sentence. " * 50

    output1, output2 = test_func(
        input1=input_workload,
        input2=input_workload,
        input3=input_workload,
    )

    output1.get()
    output2.get()


if __name__ == "__main__":
    latency = vm.profile(main)
    print(latency)
