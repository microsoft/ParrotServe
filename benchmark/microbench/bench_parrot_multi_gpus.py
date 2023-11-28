# This benchmark doesn't include the part of starting servers.
# Please manually do this by:
#   bash sample_configs/launch/launch_4_vicuna_7b.sh

import parrot as P
import asyncio
import time


def bench_4_7b_models():
    vm = P.VirtualMachine(
        os_http_addr="http://localhost:9000",
        mode="debug",
    )

    @P.semantic_function()
    def test_func(
        input: P.Input,
        output: P.Output(
            sampling_config=P.SamplingConfig(
                max_gen_length=50, ignore_tokenizer_eos=True
            ),
        ),
    ):
        """This is a test function {{input}}. {{output}}"""

    async def main():
        input = P.variable()
        call1 = test_func(input)
        call2 = test_func(input)
        call3 = test_func(input)
        call4 = test_func(input)

        time.sleep(5)  # Ensure ready

        st = time.perf_counter_ns()

        input.set("Hello")

        gets = []
        for call in [call1, call2, call3, call4]:
            gets.append(call.aget())
        await asyncio.wait(gets)

        et = time.perf_counter_ns()

        print(f"Total time: {(et - st) / 1e6} ms")

    vm.run(main())


if __name__ == "__main__":
    bench_4_7b_models()
