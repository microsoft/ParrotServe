from parrot.serve.scheduler import TaskCreator, GlobalScheduler, GlobalSchedulerConfig
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.context_manager import ServeCoreContextManager
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.graph import (
    RequestChain,
    ConstantFill,
    PlaceholderGen,
    ComputeGraph,
    PerformanceCriteria,
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
            ],
            metadata=SemanticCallMetadata(
                session_id=0,
                models=[],
                model_type="token_id",
                remove_pure_fill=True,
            ),
        )
        var_mgr.create_vars_for_request(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        task = task_creator.create_task(
            request_chain.completion_chains[0], PerformanceCriteria.THROUGHPUT
        )
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

    scheduler.schedule()

    # Expect results: all tasks go to the same engine


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
            ],
            metadata=SemanticCallMetadata(
                session_id=0,
                models=[],
                model_type="token_id",
                remove_pure_fill=True,
            ),
        )
        var_mgr.create_vars_for_request(session_id, request_chain)
        graph.insert_and_update_request_chain(request_chain)
        task = task_creator.create_task(
            request_chain.completion_chains[0], PerformanceCriteria.LATENCY
        )
        task.tokenize_chain(tokenizers_wrapper)
        scheduler.submit_task(task)

    scheduler.schedule()

    # Expect results: 4 tasks engine0, 4 tasks engine1


if __name__ == "__main__":
    test_default_policy_throughput()
    test_default_policy_latency()
