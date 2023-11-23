import parrot as P

from parrot.os.process.process import Process
from parrot.os.tokenizer import Tokenizer
from parrot.os.memory.mem_space import MemorySpace
from parrot.os.engine import ExecutionEngine
from parrot.os.process.thread import Thread
from parrot.engine.config import EngineConfig
from parrot.os.thread_dispatcher import DispatcherConfig, ThreadDispatcher


@P.function()
def test(a: P.Input, b: P.Input, c: P.Output):
    """This {{b}} is a test {{a}} function {{c}}"""


def test_default_policy():
    dispatcher_config = DispatcherConfig(
        dag_aware=False,
        app_fifo=False,
        max_queue_size=1024,
    )

    # 4 identical engines
    engine_config = EngineConfig()
    engines = {i: ExecutionEngine(engine_id=i, config=engine_config) for i in range(4)}

    # init dispatcher
    dispatcher = ThreadDispatcher(config=dispatcher_config, engines=engines)
    mem_space = MemorySpace()
    tokenizer = Tokenizer()
    process = Process(
        pid=0,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        memory_space=mem_space,
    )

    # 8 threads
    call = test("a", "b")
    for i in range(8):
        thread = Thread(tid=i, process=process, call=call, context_id=0)
        dispatcher.push_thread(thread)

    # by default, the dispatcher will dispatch threads averagely to all engines
    dispatched_threads = dispatcher.dispatch()

    assert len(dispatched_threads) == 8


def test_dag_aware_policy():
    dispatcher_config = DispatcherConfig(
        dag_aware=True,
        app_fifo=False,
        max_queue_size=1024,
    )

    # 4 identical engines
    engine_config = EngineConfig()
    engines = {i: ExecutionEngine(engine_id=i, config=engine_config) for i in range(4)}

    # init dispatcher
    dispatcher = ThreadDispatcher(config=dispatcher_config, engines=engines)
    mem_space = MemorySpace()
    tokenizer = Tokenizer()
    process = Process(
        pid=0,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        memory_space=mem_space,
    )

    # 8 threads
    call = test("a", "b")
    for i in range(8):
        thread = Thread(tid=i, process=process, call=call, context_id=0)
        dispatcher.push_thread(thread)

    # by default, the dispatcher will dispatch threads averagely to all engines
    dispatched_threads = dispatcher.dispatch()

    assert len(dispatched_threads) == 8


if __name__ == "__main__":
    test_default_policy()
