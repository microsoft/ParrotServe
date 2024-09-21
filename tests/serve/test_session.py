import time
import pytest
import asyncio

from parrot.exceptions import ParrotCoreUserError

from parrot.serve.session_manager import SessionManager
from parrot.serve.scheduler import TaskCreator, GlobalScheduler, GlobalSchedulerConfig
from parrot.serve.prefix_matcher import PrefixMatcher
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.context_manager import ServeCoreContextManager
from parrot.serve.engine_manager import EngineManager
from parrot.serve.session.graph_executor import GraphExecutor
from parrot.serve.backend_repr import ExecutionEngine

from parrot.testing.localhost_server_daemon import fake_engine_server
from parrot.testing.fake_engine_server import engine_config

from parrot.serve.graph import (
    RequestChain,
    ComputeGraph,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    PerformanceCriteria,
    activate_producer,
)
from parrot.serve.graph.call_request import SemanticFunctionParameter


def test_session_manager():
    scheduler_config = GlobalSchedulerConfig()
    prefix_matcher = PrefixMatcher()
    var_mgr = SemanticVariableManager(666)
    tokenizers_wrapper = TokenizersWrapper()
    context_mgr = ServeCoreContextManager()
    engine_mgr = EngineManager(
        tokenizers_wrapper=tokenizers_wrapper,
        context_mgr=context_mgr,
        engine_heartbeat_timeout=666,
    )
    task_creator = TaskCreator()
    scheduler = GlobalScheduler(scheduler_config, engine_mgr, context_mgr)

    session_mgr = SessionManager(
        life_span=10,
        prefix_matcher=prefix_matcher,
        task_creator=task_creator,
        scheduler=scheduler,
        var_mgr=var_mgr,
        engine_mgr=engine_mgr,
        context_mgr=context_mgr,
        tokenizers_wrapper=tokenizers_wrapper,
    )

    # Test session registration
    session_id = session_mgr.register_session()

    session = session_mgr.get_session(session_id)
    assert session.session_id == session_id

    # Test session expiration
    time.sleep(11)
    session_mgr.check_running_sessions()

    with pytest.raises(ParrotCoreUserError):
        session_mgr.check_session_status(session_id)


def test_graph_executor():
    session_id = 0

    task_creator = TaskCreator()
    scheduler_config = GlobalSchedulerConfig()
    var_mgr = SemanticVariableManager(666)
    tokenizers_wrapper = TokenizersWrapper()
    context_mgr = ServeCoreContextManager()
    engine_mgr = EngineManager(
        tokenizers_wrapper=tokenizers_wrapper,
        context_mgr=context_mgr,
        engine_heartbeat_timeout=666,
    )
    task_creator = TaskCreator()
    scheduler = GlobalScheduler(scheduler_config, engine_mgr, context_mgr)
    executor = GraphExecutor(
        session_id=session_id,
        task_creator=task_creator,
        scheduler=scheduler,
        engine_mgr=engine_mgr,
        context_mgr=context_mgr,
        tokenizers_wrapper=tokenizers_wrapper,
    )

    var_mgr.register_local_var_space(session_id)
    in_var = var_mgr.create_var(session_id, "in_var")

    request = RequestChain.from_nodes(
        nodes=[
            ConstantFill("Hello world, I'm a prefix."),
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name="a", var_id=in_var.id, is_output=False
                )
            ),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(name="b", is_output=True)
            ),
        ]
    )

    var_mgr.create_vars_for_semantic_request_chain(session_id, request)

    engine_mgr.register_engine(engine_config)

    async def main():
        executor.add_request(request)
        activate_producer(request.comp_chains[0], PerformanceCriteria.LATENCY)
        await asyncio.sleep(1)
        in_var.set("This is a test value.")
        await asyncio.sleep(0.1)
        scheduler.schedule()
        await asyncio.sleep(5)

    with fake_engine_server():
        asyncio.run(main())


if __name__ == "__main__":
    # test_session_manager()
    test_graph_executor()
