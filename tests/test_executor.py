"""This test requires a running backend server.
Use `python3 -m parrot.testing.fake_server` to start a fake server.
"""

import parrot as P


def init():
    vm = P.VirtualMachine()
    vm.init()
    vm.register_tokenizer("facebook/opt-13b")
    vm.register_engine(
        "test",
        host="localhost",
        port=8888,
        tokenizer="facebook/opt-13b",
    )
    return vm


def test_single_function_call():
    vm = init()

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    # Execute
    async def main():
        c = test("Apple", "Banana")

        # await asyncio.sleep(1)  # Simulate a long running task

        print(await c.get())

    vm.run(main(), timeit=True)


def test_pipeline_call():
    vm = init()

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    # Execute
    async def main():
        c = test("Apple", "Banana")
        d = test("Banana", c)

        # e.assign(await c.get())

        # await asyncio.sleep(1)  # Simulate a long running task

        print(await d.get())

    vm.run(main(), timeit=True)


def test_dag():
    vm = init()

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    # Execute
    async def main():
        a = "Apple"
        b = "Banana"
        c = test(a, b)
        d = test(b, c)
        e = test(a, c)
        f = test(c, d)
        g = test(e, f)

        # e.assign(await c.get())

        # await asyncio.sleep(1)  # Simulate a long running task

        print(await g.get())

    vm.run(main(), timeit=True)


def test_shared_context():
    vm = init()

    @P.function(caching_prefix=False)
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    ctx = P.shared_context(engine_name="test")

    async def main():
        ctx.fill("This is a test context.")

        a = "Apple"
        b = "Banana"

        with ctx.open("r") as handler:
            c = handler.call(test, a, b)

        print(await c.get())

        with ctx.open("w") as handler:
            d = handler.call(test, b, c)

        print(await d.get())

        with ctx.open("r") as handler:
            e = handler.call(test, a, c)

        print(await e.get())

    vm.run(main(), timeit=False)


if __name__ == "__main__":
    test_single_function_call()
    test_pipeline_call()
    test_dag()
    test_shared_context()
