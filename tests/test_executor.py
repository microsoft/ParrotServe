"""This test requires a running backend server.
Use `python3 -m parrot.testing.fake_server` to start a fake server.
"""

import time
from parrot import P, env


def init():
    env.register_tokenizer("facebook/opt-13b")
    env.register_engine(
        "test",
        host="localhost",
        port=8888,
        tokenizer="facebook/opt-13b",
    )


def test_execute_single_function_call():
    # Initialize
    init()

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    # Execute
    async def main():
        a = P.placeholder("a")
        b = P.placeholder("b")
        c = P.placeholder("c")

        a.assign("Apple")
        b.assign("Banana")
        test(a, b, c)

        # await asyncio.sleep(1)  # Simulate a long running task

        print(await c.get())

    st = time.perf_counter_ns()
    env.parrot_run_aysnc(main())
    ed = time.perf_counter_ns()
    print("Time Used: ", (ed - st) / 1e9)


def test_pipeline_call():
    # Initialize
    init()

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    # Execute
    async def main():
        a = P.placeholder("a")
        b = P.placeholder("b")
        c = P.placeholder("c")
        d = P.placeholder("d")
        # e = P.placeholder("e")

        a.assign("Apple")
        b.assign("Banana")
        test(a, b, c)
        test(b, c, d)

        # e.assign(await c.get())

        # await asyncio.sleep(1)  # Simulate a long running task

        print(await d.get())

    st = time.perf_counter_ns()
    env.parrot_run_aysnc(main())
    ed = time.perf_counter_ns()
    print("Time Used: ", (ed - st) / 1e9)


def test_dag():
    # Initialize
    init()

    @P.function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    # Execute
    async def main():
        a = P.placeholder("a")
        b = P.placeholder("b")
        c = P.placeholder("c")
        d = P.placeholder("d")
        e = P.placeholder("e")
        f = P.placeholder("f")
        g = P.placeholder("g")

        a.assign("Apple")
        b.assign("Banana")
        test(a, b, c)
        test(b, c, d)
        test(a, c, e)
        test(c, d, f)
        test(e, f, g)

        # e.assign(await c.get())

        # await asyncio.sleep(1)  # Simulate a long running task

        print(await g.get())

    st = time.perf_counter_ns()
    env.parrot_run_aysnc(main())
    ed = time.perf_counter_ns()
    print("Time Used: ", (ed - st) / 1e9)


if __name__ == "__main__":
    test_execute_single_function_call()
    # test_pipeline_call()
    # test_dag()
