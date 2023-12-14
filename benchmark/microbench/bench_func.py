import time
import parrot as P
from parrot.testing.localhost_server_daemon import fake_os_server
from parrot.utils import cprofile

def def_func():
    with fake_os_server():
        vm = P.VirtualMachine(os_http_addr="http://localhost:9000")
        st = time.perf_counter_ns()
        output_len = 32


        func = vm.define_function(
            func_name="chat",
            func_body="{{input}}{{output}}",
            params=[
                P.Parameter("input", P.ParamType.INPUT_LOC),
                P.Parameter("output", P.ParamType.OUTPUT_LOC, 
                            sampling_config=P.SamplingConfig(
                                max_gen_length=output_len,
                                ignore_tokenizer_eos=True,
                            ),  
                            dispatch_annotation=P.DispatchAnnotation(
                                requests_num_upperbound=32,
                            ),
                        ),
            ],
            cache_prefix=False,
        )

        ed = time.perf_counter_ns()
        print("define function time: ", (ed - st) / 1e9)


def call_func():
    with fake_os_server():
        vm = P.VirtualMachine(os_http_addr="http://localhost:9000")
        output_len = 32
        func = vm.define_function(
            func_name="chat",
            func_body="{{input}}{{output}}",
            params=[
                P.Parameter("input", P.ParamType.INPUT_LOC),
                P.Parameter("output", P.ParamType.OUTPUT_LOC, 
                            sampling_config=P.SamplingConfig(
                                max_gen_length=output_len,
                                ignore_tokenizer_eos=True,
                            ),  
                            dispatch_annotation=P.DispatchAnnotation(
                                requests_num_upperbound=32,
                            ),
                        ),
            ],
            cache_prefix=False,
        )
        st = time.perf_counter_ns()
        call = func("Test " * 512)
        ed = time.perf_counter_ns()
        print("call function time: ", (ed - st) / 1e9)


if __name__ == "__main__":
    for _ in range(10):
        # def_func()
        call_func()