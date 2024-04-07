import time
import pytest

from parrot.exceptions import ParrotCoreUserError

from parrot.serve.session_manager import SessionManager
from parrot.serve.scheduler import GlobalScheduler, GlobalSchedulerConfig
from parrot.serve.prefix_matcher import PrefixMatcher
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.context_manager import ServeCoreContextManager
from parrot.serve.session_manager import SessionManager
from parrot.serve.engine_manager import EngineManager


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
    scheduler = GlobalScheduler(scheduler_config, engine_mgr, context_mgr)

    session_mgr = SessionManager(
        life_span=10,
        prefix_matcher=prefix_matcher,
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


if __name__ == "__main__":
    test_session_manager()
