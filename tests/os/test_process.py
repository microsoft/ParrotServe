"""This test requires a running fake backend server.

Use `python3 -m parrot.testing.fake_engine_server` to start a fake backend 
server (without connect with OS!).
"""

import asyncio
import parrot as P

from parrot.testing.fake_engine_server import engine_name, engine_config

from parrot.os.process.process import Process
from parrot.os.memory.mem_space import MemorySpace
from parrot.os.engine import ExecutionEngine
from parrot.os.thread_dispatcher import ThreadDispatcher
from parrot.os.tokenizer import Tokenizer


def init():
    mem_space = MemorySpace()
    tokenizer = Tokenizer()
    engine_id = 0
    engine = ExecutionEngine(engine_id, engine_name, engine_config)
    dispatcher = ThreadDispatcher({engine_id: engine})

    pid = 0
    process = Process(pid, dispatcher, mem_space, tokenizer)
    return process


def test_single_call():
    proc = init()

    async def main():
        @P.function()
        def test(a: P.Input, b: P.Input, c: P.Output):
            """This {{b}} is a test {{a}} function {{c}}"""

        # Without VM, it will return the call
        call = test("Apple", "Banana")
        future_id = call.output_futures[0].id

        proc.execute_call(call)

        content = await proc.placeholders_map[future_id].get()

        print(content)

    asyncio.run(main())


if __name__ == "__main__":
    test_single_call()
