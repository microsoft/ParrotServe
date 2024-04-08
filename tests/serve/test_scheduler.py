from typing import List
from parrot.serve.scheduler import TaskCreator, GlobalScheduler, GlobalSchedulerConfig
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.context_manager import ServeCoreContextManager
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.graph import (
    RequestChain,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    ComputeGraph,
    PerformanceCriteria,
    activate_completion_chain,
    SemanticVariable,
)
from parrot.serve.graph.request import SemanticCallMetadata, RequestPlaceholder
from parrot.engine.config import EngineConfig
from parrot.serve.engine_manager import EngineManager


def test_default_policy_throughput():
    scheduler_cfg = GlobalSchedulerConfig(
        app_fifo=False,
        graph_group=False,
        ctx_group=False,
        ctx_aware=False,
        max_queue_size=1024,
    )

    graph = ComputeGraph()
    tokenizers_wrapper = TokenizersWrapper()
    context_mgr = ServeCoreContextManager()
    engine_mgr = EngineManager(
        tokenizers_wrapper=tokenizers_wrapper,
        context_mgr=context_mgr,
        engine_heartbeat_timeout=666,
    )

    scheduler = GlobalScheduler(
        config=scheduler_cfg,
        engine_mgr=engine_mgr,
        context_mgr=context_mgr,
    )
    task_creator = TaskCreator()

    # Register 4 identical engines
    engine_config = EngineConfig(tokenizer="hf-internal-testing/llama-tokenizer")
    for _ in range(4):
        engine_mgr.register_engine(engine_config)

    var_mgr = SemanticVariableManager(666)
    session_id = 0
    var_mgr.register_local_var_space(session_id)

    # 8 identical tasks
    for _ in range(8):
        request_chain = RequestChain.from_nodes(
            nodes=[
                ConstantFill("This is a test "),
                PlaceholderGen(
                    placeholder=RequestPlaceholder(name="a", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_request(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        activate_completion_chain(comp_chain, PerformanceCriteria.THROUGHPUT)
        task = task_creator.create_task(comp_chain)
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

    scheduler.schedule()

    # Expected results: all tasks go to the same engine


def test_default_policy_latency():
    scheduler_cfg = GlobalSchedulerConfig(
        app_fifo=False,
        graph_group=False,
        ctx_group=False,
        ctx_aware=False,
        max_queue_size=1024,
    )

    graph = ComputeGraph()
    tokenizers_wrapper = TokenizersWrapper()
    context_mgr = ServeCoreContextManager()
    engine_mgr = EngineManager(
        tokenizers_wrapper=tokenizers_wrapper,
        context_mgr=context_mgr,
        engine_heartbeat_timeout=666,
    )

    scheduler = GlobalScheduler(
        config=scheduler_cfg,
        engine_mgr=engine_mgr,
        context_mgr=context_mgr,
    )
    task_creator = TaskCreator()

    # Register 4 identical engines
    engine_config = EngineConfig(tokenizer="hf-internal-testing/llama-tokenizer")
    for _ in range(4):
        engine_mgr.register_engine(engine_config)

    var_mgr = SemanticVariableManager(666)
    session_id = 0
    var_mgr.register_local_var_space(session_id)

    # 8 identical tasks
    for _ in range(8):
        request_chain = RequestChain.from_nodes(
            nodes=[
                ConstantFill("This is a test "),
                PlaceholderGen(
                    placeholder=RequestPlaceholder(name="a", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_request(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        activate_completion_chain(comp_chain, PerformanceCriteria.LATENCY)
        task = task_creator.create_task(comp_chain)
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

    scheduler.schedule()

    # Expected results: 4 tasks engine0, 4 tasks engine1


def test_app_fifo():
    scheduler_cfg = GlobalSchedulerConfig(
        app_fifo=True,
        graph_group=False,
        ctx_group=False,
        ctx_aware=False,
        max_queue_size=1024,
    )

    graph = ComputeGraph()
    tokenizers_wrapper = TokenizersWrapper()
    context_mgr = ServeCoreContextManager()
    engine_mgr = EngineManager(
        tokenizers_wrapper=tokenizers_wrapper,
        context_mgr=context_mgr,
        engine_heartbeat_timeout=666,
    )

    scheduler = GlobalScheduler(
        config=scheduler_cfg,
        engine_mgr=engine_mgr,
        context_mgr=context_mgr,
    )
    task_creator = TaskCreator()

    # Register 1 engine with limited capacity
    engine_config = EngineConfig(
        tokenizer="hf-internal-testing/llama-tokenizer", tasks_capacity=1
    )
    engine_mgr.register_engine(engine_config)

    var_mgr = SemanticVariableManager(666)
    session_id = 0
    var_mgr.register_local_var_space(session_id)

    # 8 tasks. Each group of 2 tasks with A->B dependency.
    tasks = []
    out_vars: List[SemanticVariable] = []

    for _ in range(4):
        request_chain = RequestChain.from_nodes(
            nodes=[
                ConstantFill("This is a test "),
                PlaceholderGen(
                    placeholder=RequestPlaceholder(name="a", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_request(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        activate_completion_chain(comp_chain, PerformanceCriteria.LATENCY)
        task = task_creator.create_task(comp_chain)
        task.tokenize_chain(tokenizers_wrapper)
        out_vars.append(comp_chain.gen_node.sv)
        tasks.append(task)
        # First 4 tasks
        scheduler.submit_task(task)

    for i in range(4):
        # Schedule.
        # Expected result: No. i task in engine0.
        scheduler.schedule()
        # Set var as finish
        out_vars[i].set("Content0")

        request_chain = RequestChain.from_nodes(
            nodes=[
                PlaceholderFill(
                    placeholder=RequestPlaceholder(
                        name="a",
                        var_id=out_vars[i].id,
                        is_output=False,
                    )
                ),
                PlaceholderGen(
                    placeholder=RequestPlaceholder(name="b", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_request(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        activate_completion_chain(comp_chain, PerformanceCriteria.LATENCY)
        task = task_creator.create_task(comp_chain)
        task.tokenize_chain(tokenizers_wrapper)
        tasks.append(task)
        scheduler.submit_task(task)

        # Schedule again.
        # Expected result: No. i+4 task in engine 0.
        scheduler.schedule()


def test_graph_group():
    pass


def test_ctx_group():
    pass


def test_ctx_aware():
    pass


if __name__ == "__main__":
    # test_default_policy_throughput()
    # test_default_policy_latency()
    test_app_fifo()
