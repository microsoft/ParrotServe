# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from abc import ABC, abstractmethod
from typing import Dict, AsyncGenerator
import asyncio
import time
import threading

from parrot.constants import ENGINE_LOOP_INTERVAL
from parrot.protocol.internal.layer_apis import register_engine, engine_heartbeat
from parrot.protocol.internal.runtime_info import EngineRuntimeInfo
from parrot.utils import get_logger, set_random_seed

from .config import EngineConfig


logger = get_logger("LLMEngine")


class LLMEngine(ABC):
    """Base class for all LLM engines. It provides a minimal interface for
    LLM engines."""

    def __init__(self, engine_config: Dict, connect_to_core: bool = True):
        # Set global random seed
        set_random_seed(engine_config["random_seed"])

        self.engine_config = EngineConfig.from_dict(engine_config)

        self.connect_to_core = connect_to_core
        if self.connect_to_core:
            assert (
                "serve_core" in engine_config
            ), 'If connect_to_core is True, "serve core" config must be provided.'
            core_config = engine_config["serve_core"]

            self.serve_core_http_address = (
                f"http://{core_config['host']}:{core_config['port']}"
            )

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_daemon, daemon=True
        )

    def _register_engine(self, engine_config: EngineConfig):
        """Register engine to ServeCore."""

        if self.connect_to_core:
            resp = register_engine(
                http_addr=self.serve_core_http_address,
                engine_config=engine_config,
            )
            self.engine_id = resp.engine_id
        else:
            self.engine_id = 0

    # ---------- Public APIs ----------

    @abstractmethod
    async def fill(self, payload: Dict) -> Dict:
        """Fill API.

        Args:
            payload: Dict[str, Any]. The payload of the fill API.

        Returns:
            Dict. The response of the fill API.
        """
        ...

    @abstractmethod
    async def generate(self, payload: Dict) -> Dict:
        """Generate API.

        Args:
            payload: Dict[str, Any]. The payload of the generate API.

        Returns:
            Dict. The response of the generate API.
        """
        ...

    @abstractmethod
    def generate_stream(self, payload: Dict) -> AsyncGenerator:
        """Generate stream API.

        Args:
            payload: Dict[str, Any]. The payload of the generate stream API.

        Returns:
            The generator of the generate stream API.
        """
        raise NotImplementedError

    @abstractmethod
    async def free_context(self, payload: Dict) -> Dict:
        """Free context API.

        Args:
            payload: Dict[str, Any]. The payload of the free context API.

        Returns:
            Dict. The response of the free context API.
        """
        ...

    @abstractmethod
    def get_runtime_info(self, profile: bool) -> EngineRuntimeInfo:
        """Get runtime info of this engine.

        Return: EngineRuntimeInfo."""
        ...

    @abstractmethod
    async def engine_iter(self):
        """The function executed in the every iteration of the engine loop."""
        ...

    # Implemented methods

    def heartbeat(self):
        """Heartbeat sent to ServeCore.

        Return: num_cached_tokens, cached_tokens_size. num_running_jobs."""

        if not self.connect_to_core:
            return

        logger.debug(
            f"Heartbeat sent to ServeCore (address={self.serve_core_http_address})."
        )

        engine_name = self.engine_config.engine_name
        engine_id = self.engine_id
        engine_runtime_info = self.get_runtime_info(profile=False)  # Performance

        logger.debug(
            f"Engine {engine_name} (id={engine_id}) heartbeat sent. "
            "Runtime info: \n" + engine_runtime_info.display()
        )

        resp = engine_heartbeat(
            http_addr=self.serve_core_http_address,
            engine_id=engine_id,
            engine_name=engine_name,
            runtime_info=engine_runtime_info,
        )

    def _heartbeat_daemon(self):
        """Loop for heartbeat."""

        while True:
            self.heartbeat()  # Send heartbeat to ServeCore
            time.sleep(self.engine_config.heartbeat_interval)

    async def engine_loop(self):
        """Engine loop, execute jobs token by token.

        For some types of engines, e.g. OpenAI engine, the engine loop is empty loop.
        """

        # Start heartbeat daemon
        self.heartbeat_thread.start()

        while True:
            await asyncio.sleep(ENGINE_LOOP_INTERVAL)
            await self.engine_iter()
