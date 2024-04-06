import json
import time


from parrot.serve.backend_repr import ExecutionEngine, LanguageModel
from parrot.engine.config import EngineConfig
from parrot.serve.tokenizer_wrapper import TokenizersWrapper
from parrot.serve.engine_manager import EngineManager
from parrot.testing.get_configs import get_sample_engine_config_path


def test_engine_manager():

    tokenizers_wrapper = TokenizersWrapper()
    engine_manager = EngineManager(
        tokenizers_wrapper=tokenizers_wrapper, engine_heartbeat_timeout=5
    )
    config_path = get_sample_engine_config_path("opt-13b.json")

    with open(config_path, "r") as f:
        engine_config = EngineConfig.from_dict(json.load(f))

    # Test engine registration
    engine_id = engine_manager.register_engine(engine_config)

    engine = engine_manager.get_engine(engine_id)
    print(engine.model)
    assert engine.engine_id == engine_id

    # Test engine expiration
    time.sleep(6)
    engine_manager.update_expired_engines()
    engine_manager.sweep_not_running_engines()

    print(engine_manager.engines, engine_manager.models)


if __name__ == "__main__":
    test_engine_manager()
