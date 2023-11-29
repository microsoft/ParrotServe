import parrot as P

vm = P.VirtualMachine("http://localhost:9000", mode="release")


@P.semantic_function(cache_prefix=False)
def test(
    input: P.Input,
    output: P.Output(
        sampling_config=P.SamplingConfig(
            max_gen_length=50,
            ignore_tokenizer_eos=True,
        ),
    ),
):
    """{{input}}{{output}}"""


with vm.running_scope():
    output = test("hello")
    print(output.get())
