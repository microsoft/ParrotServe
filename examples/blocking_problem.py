import parrot as P

vm = P.VirtualMachine(
    os_http_addr="http://localhost:9000",
    mode="release",
)


@P.function(formatter=P.allowing_newline)
def test(a: P.Input, b: P.Output(max_gen_length=100)):
    """
    This is a {{a}} test function: {{b}}.
    """


def main():
    results = []
    for i in range(50):
        result = test(str(i))
        results.append(result)

    for result in results:
        print(result.get())


vm.run(main, timeit=True)
