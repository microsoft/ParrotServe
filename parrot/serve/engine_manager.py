# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Optional, Tuple

from parrot.utils import RecyclePool, get_logger, time_counter_in_nanoseconds
from parrot.constants import ENGINE_HEARTBEAT_TIMEOUT_INTERVAL
from parrot.engine.config import EngineConfig

from parrot.serve.backend_repr import ExecutionEngine, LanguageModel

from .tokenizer_wrapper import TokenizersWrapper

logger = get_logger("EngineManager")


class EngineManager:
    """Manage engines in a cluster level.

    The aim is to manage LLM engines ranging from different types, models and GPUs inside the cloud service.

    Note that engines may connect/disconnect to the cluster at any time.
    """

    def __init__(self, tokenizers_wrapper: TokenizersWrapper) -> None:
        # engine_id -> engine
        self.engines: Dict[int, ExecutionEngine] = {}

        # engine_id -> last_seen_time
        self.engine_last_seen_time: Dict[int, int] = {}

        # model_name -> model
        self.models: Dict[str, LanguageModel] = {}
        self.models_ref_counter: Dict[str, int] = {}
        self.id_pool: RecyclePool = ()

        self.tokenizers_wrapper = tokenizers_wrapper

    def _register_model(self, model: LanguageModel) -> None:
        self.models[model.model_name] = model
        self.tokenizers_wrapper.register_tokenizer(model.tokenizer_name)
        logger.debug(f"Model {model.model_name} registered.")

    def _remove_model(self, model_name: str) -> None:
        self.models_ref_counter[model_name] -= 1
        if self.models_ref_counter[model_name] == 0:
            model = self.models.pop(model_name)
            self.tokenizers_wrapper.remove_tokenizer(model.tokenizer_name)
            logger.debug(f"Model {model_name} removed.")

    def _remove_engine(self, engine_id: int) -> None:
        engine = self.engines.pop(engine_id)
        self.engine_last_seen_time.pop(engine_id)
        self.id_pool.free(engine_id)
        self._remove_model(engine.model.model_name)
        logger.debug(f"Engine {engine.name} (id={engine_id}) is removed.")

    # ---------- Methods for Executor ----------

    def raise_exception(self, engine_id: int, exception: Exception) -> None:
        """Raise an exception in the engine.

        Args:
            engine_id (int): The engine ID.
            exception (Exception): The exception to be raised.
        """

        engine = self.engines[engine_id]
        engine.mark_bad(exception=exception)

    # ---------- Methods for Core ----------

    async def register_engine(self, engine_config: EngineConfig) -> int:
        """Register an engine to the cluster.

        Args:
            engine_config (EngineConfig): The configuration of the engine.

        Returns:
            int: The engine ID.
        """

        # Register the model
        model = LanguageModel.from_engine_config(engine_config)
        self._register_model(model)

        # Register the engine
        engine_id = self.id_pool.allocate()
        engine = ExecutionEngine(engine_id=engine_id, config=engine_config, model=model)
        self.engines[engine_id] = engine
        self.engine_last_seen_time[engine_id] = time_counter_in_nanoseconds()

        logger.debug(f"Engine {engine.name} (id={engine_id}) registered.")
        return engine_id

    async def engine_heartbeat(self, engine_id: int) -> None:
        """Update the last seen time of the engine.

        Args:
            engine_id (int): The engine ID.
        """

        self.engine_last_seen_time[engine_id] = time_counter_in_nanoseconds()

    async def update_expired_engines(self) -> None:
        """If the engine is expired, update the engine status."""

        current_time = time_counter_in_nanoseconds()
        for engine_id, last_seen_time in self.engine_last_seen_time.items():
            engine = self.engines[engine_id]
            if (
                current_time - last_seen_time
                > ENGINE_HEARTBEAT_TIMEOUT_INTERVAL * 1_000_000_000
            ):
                engine.mark_dead()
                logger.debug(f"Engine {engine_id} is expired.")

    async def sweep_dead_engines(self) -> None:
        """Sweep the dead/bad engines."""

        for engine_id, engine in self.engines.items():
            if engine.not_running:
                self.engines.pop(engine_id)
                self.engine_last_seen_time.pop(engine_id)
                self.id_pool.free(engine_id)
                logger.debug(f"Engine {engine.name} (id={engine_id}) is removed.")
                continue
