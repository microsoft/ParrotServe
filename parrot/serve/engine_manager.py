# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List, Optional, Tuple

from parrot.exceptions import ParrotCoreUserError, parrot_assert
from parrot.utils import RecyclePool, get_logger, time_counter_in_nanoseconds
from parrot.protocol.internal.runtime_info import EngineRuntimeInfo
from parrot.engine.config import EngineConfig
from parrot.protocol.internal.layer_apis import ping_engine

from parrot.serve.backend_repr import ExecutionEngine, LanguageModel, EngineStatus

from .tokenizer_wrapper import TokenizersWrapper
from .context_manager import ServeCoreContextManager


logger = get_logger("EngineManager")


class EngineManager:
    """Manage engines in a cluster level.

    The aim is to manage LLM engines ranging from different types, models and GPUs inside the cloud service.

    Note that engines may connect/disconnect to the cluster at any time.
    """

    def __init__(
        self,
        tokenizers_wrapper: TokenizersWrapper,
        context_mgr: ServeCoreContextManager,
        engine_heartbeat_timeout: int,
    ) -> None:
        # engine_id -> engine
        self.engines: Dict[int, ExecutionEngine] = {}

        # engine_id -> last_seen_time
        self.engine_last_seen_time: Dict[int, int] = {}

        # model_name -> model
        self.models: Dict[str, LanguageModel] = {}
        self.models_ref_counter: Dict[str, int] = {}
        self.engine_id_pool = RecyclePool()

        # ---------- Global Components ----------
        self.context_mgr = context_mgr
        self.tokenizers_wrapper = tokenizers_wrapper

        self.engine_heartbeat_timeout = engine_heartbeat_timeout

    def _register_model(self, model: LanguageModel) -> LanguageModel:
        if model.model_name in self.models:
            self.models_ref_counter[model.model_name] += 1
            return self.models[model.model_name]

        self.models[model.model_name] = model
        self.models_ref_counter[model.model_name] = 1
        self.tokenizers_wrapper.register_tokenizer(model.tokenizer_name)
        logger.debug(f"Model {model.model_name} registered.")
        return model

    def _remove_model(self, model_name: str) -> None:
        self.models_ref_counter[model_name] -= 1
        if self.models_ref_counter[model_name] == 0:
            model = self.models.pop(model_name)
            self.tokenizers_wrapper.remove_tokenizer(model.tokenizer_name)
            logger.debug(f"Model {model_name} removed.")

    def _remove_engine(self, engine_id: int) -> None:
        engine = self.engines.pop(engine_id)

        self._remove_model(engine.model_name)

        self.engine_last_seen_time.pop(engine_id)
        self.engine_id_pool.free(engine_id)

        self.context_mgr.remove_engine_prefix_cache(engine_id)

        logger.debug(f"Engine {engine.name} (id={engine_id}) is removed.")

    # ---------- Methods for Executor ----------

    def raise_exception(self, engine_id: int, exception: Exception) -> None:
        """Raise an exception in the engine.

        Args:
            engine_id: int. The engine ID.
            exception: Exception. The exception to be raised.
        """

        engine = self.engines[engine_id]
        engine.mark_bad(exception)

    # ---------- Methods for Global Scheduler ----------

    def get_live_engines(self) -> List[ExecutionEngine]:
        """Get all live engines."""

        return [engine for engine in self.engines.values() if engine.is_running]

    # ---------- Methods for Core ----------

    def register_engine(self, engine_config: EngineConfig) -> int:
        """Register an engine to the cluster.

        Args:
            engine_config: EngineConfig. The configuration of the engine.

        Returns:
            int: The engine ID.
        """

        # Register the model
        model = LanguageModel.from_engine_config(engine_config)
        model = self._register_model(model)

        # Register the engine
        engine_id = self.engine_id_pool.allocate()
        engine = ExecutionEngine(engine_id=engine_id, config=engine_config, model=model)

        self.engines[engine_id] = engine
        self.engine_last_seen_time[engine_id] = time_counter_in_nanoseconds()

        # Register engine prefix cache
        self.context_mgr.register_engine_prefix_cache(engine_id=engine_id)

        logger.debug(f"Engine {engine.name} (id={engine_id}) registered.")
        return engine_id

    def engine_heartbeat(
        self, engine_id: int, engine_runtime_info: EngineRuntimeInfo
    ) -> None:
        """Update the last seen time of the engine.

        Args:
            engine_id: int. The engine ID.
        """

        if engine_id not in self.engines:
            raise ParrotCoreUserError(f"Engine {engine_id} not found.")

        engine = self.engines[engine_id]
        engine.update_realtime_runtime_info(engine_runtime_info)

        self.engine_last_seen_time[engine_id] = time_counter_in_nanoseconds()

    def get_engine(self, engine_id: int) -> ExecutionEngine:
        """Get the ExecutionEngine by engine ID.

        Args:
            engine_id: int. The engine ID.

        Returns:
            ExecutionEngine: The engine.
        """

        parrot_assert(engine_id in self.engines, f"Engine {engine_id} not found.")
        return self.engines[engine_id]

    def update_expired_engines(self) -> None:
        """If the engine is expired, update the engine status."""

        current_time = time_counter_in_nanoseconds()
        for engine_id, last_seen_time in self.engine_last_seen_time.items():
            engine = self.engines[engine_id]
            if (
                current_time - last_seen_time
                > self.engine_heartbeat_timeout * 1_000_000_000
            ):
                engine.status = EngineStatus.DEAD
                logger.debug(f"Engine {engine_id} is expired.")

    def sweep_not_running_engines(self) -> None:
        """Sweep the dead/bad engines."""

        engines_copy = self.engines.copy()

        for engine_id, engine in engines_copy.items():
            if not engine.is_running:
                self._remove_engine(engine_id)
