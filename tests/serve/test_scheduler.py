from typing import List, Optional
from parrot.serve.scheduler import (
    CompletionTask,
    TaskCreator,
    GlobalScheduler,
    GlobalSchedulerConfig,
)
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.context_manager import ServeCoreContextManager
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.graph import (
    RequestChain,
    CompletionChain,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    ComputeGraph,
    PerformanceCriteria,
    activate_sv,
    SemanticVariable,
)
from parrot.serve.graph.call_request import (
    SemanticCallMetadata,
    SemanticFunctionParameter,
)
from parrot.engine.config import EngineConfig
from parrot.serve.engine_manager import EngineManager
from parrot.serve.graph.visualize_utils import view_graph


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
                    parameter=SemanticFunctionParameter(name="a", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        activate_sv(comp_chain.gen_node.sv, PerformanceCriteria.THROUGHPUT)
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
                    parameter=SemanticFunctionParameter(name="a", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        activate_sv(comp_chain.gen_node.sv, PerformanceCriteria.LATENCY)
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
    first_batch_tasks: List[CompletionTask] = []
    second_batch_chains: List[CompletionChain] = []
    out_vars: List[SemanticVariable] = []

    for i in range(4):
        request_chain1 = RequestChain.from_nodes(
            nodes=[
                ConstantFill("This is a test "),
                PlaceholderGen(
                    parameter=SemanticFunctionParameter(name="a", is_output=True)
                ),
            ]
        )

        var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain1)
        graph.insert_and_update_request_chain(request_chain1)
        comp_chain1 = request_chain1.comp_chains[0]
        out_vars.append(comp_chain1.gen_node.sv)

        request_chain2 = RequestChain.from_nodes(
            nodes=[
                PlaceholderFill(
                    parameter=SemanticFunctionParameter(
                        name="a",
                        var_id=out_vars[i].id,
                        is_output=False,
                    )
                ),
                PlaceholderGen(
                    parameter=SemanticFunctionParameter(name="b", is_output=True)
                ),
            ]
        )

        var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain2)
        graph.insert_and_update_request_chain(request_chain2)
        comp_chain2 = request_chain2.comp_chains[0]
        activate_sv(comp_chain2.gen_node.sv, PerformanceCriteria.LATENCY)

        task1 = task_creator.create_task(comp_chain1)
        task1.tokenize_chain(tokenizers_wrapper)
        first_batch_tasks.append(task1)
        second_batch_chains.append(comp_chain2)

        scheduler.submit_task(task1)

    # view_graph(graph)

    for i in range(4):
        # Schedule.
        # Expected result: No. i task in engine0.
        scheduler.schedule()
        # Set var as finish
        out_vars[i].set("Content0")
        assert first_batch_tasks[i].is_scheduled
        first_batch_tasks[i].leave_scheduled()

        # Submit 2
        comp_chain = second_batch_chains[i]
        task = task_creator.create_task(comp_chain)
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

        # Schedule again.
        # Expected result: No. i+4 task in engine 0.
        scheduler.schedule()
        assert task.is_scheduled
        task.engine.update_servelayer_runtime_info_remove_task(task)

    # view_graph(graph)
    # 0 4 1 5 2 6 3 7


def test_graph_group():
    scheduler_cfg = GlobalSchedulerConfig(
        app_fifo=False,
        graph_group=True,
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

    # Register 1 engine
    engine_config = EngineConfig(tokenizer="hf-internal-testing/llama-tokenizer")
    engine_mgr.register_engine(engine_config)

    var_mgr = SemanticVariableManager(666)
    session_id = 0
    var_mgr.register_local_var_space(session_id)

    # In this case we construct a graph with 17 requests, with 16 requests dependent on the 17th.
    out_vars: List[SemanticVariable] = []
    chains: List[CompletionChain] = []
    for _ in range(16):
        request_chain = RequestChain.from_nodes(
            nodes=[
                ConstantFill("This is a test "),
                PlaceholderGen(
                    parameter=SemanticFunctionParameter(name="a", is_output=True)
                ),
            ]
        )

        var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        out_vars.append(comp_chain.gen_node.sv)
        chains.append(comp_chain)

    # Construct the final request
    request_chain = RequestChain.from_nodes(
        nodes=[
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name=f"a_{i}",
                    var_id=out_vars[i].id,
                    is_output=False,
                )
            )
            for i in range(16)
        ]
        + [
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="b", is_output=True)
            ),
        ]
    )
    var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)
    graph.insert_and_update_request_chain(request_chain)
    comp_chain = request_chain.comp_chains[0]
    activate_sv(comp_chain.gen_node.sv, PerformanceCriteria.LATENCY)

    # view_graph(graph)

    for i in range(16):
        task = task_creator.create_task(chains[i])
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

    scheduler.schedule()

    # Expected result: 16 tasks go to the engine 0, ignoring the task_num_upperbound as a group.


def test_ctx_group():
    scheduler_cfg = GlobalSchedulerConfig(
        app_fifo=False,
        graph_group=False,
        ctx_group=True,
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

    # 16 identical tasks
    prompts = [f"This is a test {i}" for i in range(4)]
    for i in range(16):
        request_chain = RequestChain.from_nodes(
            nodes=[
                ConstantFill(prompts[i % 4]),
                PlaceholderGen(
                    parameter=SemanticFunctionParameter(name="a", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        activate_sv(comp_chain.gen_node.sv, PerformanceCriteria.LATENCY)
        task = task_creator.create_task(comp_chain)
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

    scheduler.schedule()

    # Expected results: 0, 4, 8, 12 tasks go to engine 0, 1, 2, 3 respectively.


def test_ctx_aware():
    scheduler_cfg = GlobalSchedulerConfig(
        app_fifo=False,
        graph_group=False,
        ctx_group=False,
        ctx_aware=True,
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

    # 16 identical tasks
    prompts = [f"This is a test {i}" for i in range(4)]
    first_vars: List[SemanticVariable] = []
    for i in range(16):
        request_chain = RequestChain.from_nodes(
            nodes=[
                ConstantFill(prompts[i % 4]),
                PlaceholderGen(
                    parameter=SemanticFunctionParameter(name="a", is_output=True)
                ),
            ]
        )
        var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        comp_chain = request_chain.comp_chains[0]
        first_vars.append(comp_chain.first_node.sv)
        activate_sv(comp_chain.gen_node.sv, PerformanceCriteria.THROUGHPUT)
        task = task_creator.create_task(comp_chain)
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

    # Assign context in a round-robin manner (hacky)
    for i in range(4):
        prefix_cache = context_mgr.prefix_caches[i]
        prefix_hash = context_mgr._hash_var_id(first_vars[i].id)
        prefix_cache.cache_prefix_context(prefix_hash=prefix_hash, context_id=i)

    scheduler.schedule()

    # Expected results: 0, 4, 8, 12 tasks go to engine 0, 1, 2, 3 respectively.


if __name__ == "__main__":
    # test_default_policy_throughput()
    # test_default_policy_latency()
    # test_app_fifo()
    test_graph_group()
    # test_ctx_group()
    # test_ctx_aware()
