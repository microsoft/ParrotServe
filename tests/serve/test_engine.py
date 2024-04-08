import json
import time


from parrot.engine.config import EngineConfig
from parrot.serve.context_manager import ServeCoreContextManager
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.engine_manager import EngineManager
from parrot.testing.get_configs import get_sample_engine_config_path


def test_engine_manager():
    context_mgr = ServeCoreContextManager()
    tokenizers_wrapper = TokenizersWrapper()
    engine_mgr = EngineManager(
        tokenizers_wrapper=tokenizers_wrapper,
        context_mgr=context_mgr,
        engine_heartbeat_timeout=5,
    )
    config_path = get_sample_engine_config_path("opt-13b.json")

    with open(config_path, "r") as f:
        engine_config = EngineConfig.from_dict(json.load(f))

    # Test engine registration
    engine_id = engine_mgr.register_engine(engine_config)

    engine = engine_mgr.get_engine(engine_id)
    print(engine.model)
    assert engine.engine_id == engine_id

    # Test engine expiration
    time.sleep(6)
    engine_mgr.update_expired_engines()
    engine_mgr.sweep_not_running_engines()

    print(engine_mgr._engines, engine_mgr._models)


if __name__ == "__main__":
    test_engine_manager()
