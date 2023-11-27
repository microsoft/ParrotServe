import parrot as P

from parrot.os.process.process import Process
from parrot.os.tokenizer import Tokenizer
from parrot.os.memory.mem_space import MemorySpace
from parrot.os.engine import ExecutionEngine
from parrot.os.process.thread import Thread
from parrot.engine.config import EngineConfig
from parrot.os.thread_dispatcher import DispatcherConfig, ThreadDispatcher


@P.function()
def test(a: P.Input, b: P.Output):
    """This is a test {{a}} function {{b}}"""


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
    call = test("a")
    for i in range(8):
        thread = Thread(tid=i, process=process, call=call, context_id=0)
        dispatcher.push_thread(thread)

    dispatched_threads = dispatcher.dispatch()

    assert len(dispatched_threads) == 8

    # Expect results: dispatch threads averagely to all engines i.e. 2 threads per engine


@P.function()
def test_func_throughput(
    a: P.Input,
    b: P.Output(
        dispatch_annotation=P.DispatchAnnotation(requests_num_upperbound=64),
    ),
):
    """This is a test {{a}} function {{b}}"""


@P.function()
def test_func_latency(
    a: P.Input,
    b: P.Output(
        dispatch_annotation=P.DispatchAnnotation(requests_num_upperbound=3),
    ),
):
    """This is a test {{a}} function {{b}}"""


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

    # 16 threads, with 8 throughput threads and 8 latency threads
    call1 = test_func_throughput("a")
    call2 = test_func_latency("a")
    for i in range(8):
        thread = Thread(tid=i, process=process, call=call1, context_id=0)
        dispatcher.push_thread(thread)
    for i in range(8, 16):
        thread = Thread(tid=i, process=process, call=call2, context_id=0)
        dispatcher.push_thread(thread)

    dispatched_threads = dispatcher.dispatch()

    assert len(dispatched_threads) == 16

    # Expect results: 8 throughput threads are dispatched to the same engine,
    # and 8 latency threads are dispatched averagely to the rest 3 engines.


def test_request_wait():
    dispatcher_config = DispatcherConfig(
        dag_aware=False,
        app_fifo=False,
        max_queue_size=1024,
    )

    # 1 engine with max_threads_num=1
    engine_config = EngineConfig(max_threads_num=1)
    engines = {0: ExecutionEngine(engine_id=0, config=engine_config)}

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

    # 4 threads
    call = test("a")
    for i in range(4):
        thread = Thread(tid=i, process=process, call=call, context_id=0)
        dispatcher.push_thread(thread)

    dispatched_threads = dispatcher.dispatch()
    assert len(dispatched_threads) == 1


def test_app_fifo():
    dispatcher_config = DispatcherConfig(
        dag_aware=False,
        app_fifo=True,
        max_queue_size=1024,
    )

    # 1 engine with max_threads_num=1
    engine_config = EngineConfig(max_threads_num=1)
    engines = {0: ExecutionEngine(engine_id=0, config=engine_config)}

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

    # 8 calls, each group of 2 calls with A->B dependency.
    threads = []
    for i in range(4):
        tid1 = i
        tid2 = i + 4
        middle_node = P.future(name=f"middle_{i}")
        call1 = test("a", b=middle_node)
        call2 = test(middle_node)

        # Necessary to make DAG
        process.rewrite_call(call1)
        process.rewrite_call(call2)

        thread1 = Thread(tid=tid1, process=process, call=call1, context_id=0)
        call1.thread = thread1
        thread2 = Thread(tid=tid2, process=process, call=call2, context_id=0)
        call2.thread = thread2

        threads.append(thread1)
        threads.append(thread2)

    threads.sort(key=lambda x: x.tid)  # sort as A, A, A, A, B, B, B, B order

    for thread in threads:
        dispatcher.push_thread(thread)

    for _ in range(8):
        dispatched_threads = dispatcher.dispatch()
        assert len(dispatched_threads) == 1
        thread = dispatched_threads[0]
        thread.engine.remove_thread(thread)  # Free loc


if __name__ == "__main__":
    # test_default_policy()
    # test_dag_aware_policy()
    # test_request_wait()
    test_app_fifo()
