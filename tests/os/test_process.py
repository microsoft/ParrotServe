import asyncio
import parrot as P

from parrot.testing.fake_engine_server import engine_config
from parrot.testing.localhost_server_daemon import fake_engine_server

from Parrot.parrot.os.session.session import Process
from Parrot.parrot.os.context.context_manager import MemorySpace
from Parrot.parrot.os.engine.engine_node import ExecutionEngine
from parrot.os.thread_dispatcher import DispatcherConfig
from parrot.os.thread_dispatcher import ThreadDispatcher
from parrot.os.tokenizer import Tokenizer


def init():
    dispatcher_config = DispatcherConfig()

    mem_space = MemorySpace()
    tokenizer = Tokenizer()
    engine_id = 0
    engine = ExecutionEngine(
        engine_id=engine_id,
        config=engine_config,
        tokenizer=tokenizer,
    )
    dispatcher = ThreadDispatcher(config=dispatcher_config, engines={engine_id: engine})

    pid = 0
    process = Process(pid, dispatcher, mem_space, tokenizer)

    return dispatcher, process


def test_single_call():
    dispatcher, proc = init()

    async def main():
        @P.semantic_function()
        def test(a: P.Input, b: P.Input, c: P.Output):
            """This {{b}} is a test {{a}} function {{c}}"""

        # Without VM, it will return the call
        call = test("Apple", "Banana")
        future_id = call.output_vars[0].id

        proc.rewrite_call(call)
        thread = proc.make_thread(call)
        dispatcher.push_thread(thread)

        dispatched_threads = dispatcher.dispatch()

        assert len(dispatched_threads) == 1
        proc.execute_thread(dispatched_threads[0])

        content = await proc.placeholders_map[future_id].get()

        print(content)

    with fake_engine_server():
        asyncio.run(main())


def test_make_dag():
    dispatcher, proc = init()

    @P.semantic_function()
    def test1(a: P.Input, b: P.Output):
        """This is a test {{a}} function {{b}}"""

    @P.semantic_function()
    def test2(c: P.Input, d: P.Output):
        """This is a test {{c}} function {{d}}"""

    @P.semantic_function()
    def test3(e: P.Input, f: P.Output):
        """This is a test {{e}} function {{f}}"""

    # Without VM, it will return the call
    call1 = test1("Apple")
    b1 = call1.output_vars[0]
    call2 = test2(b1)
    b2 = call2.output_vars[0]
    call3 = test3(b2)

    proc.rewrite_call(call1)
    proc.rewrite_call(call2)
    proc.rewrite_call(call3)


if __name__ == "__main__":
    test_single_call()
    test_make_dag()
