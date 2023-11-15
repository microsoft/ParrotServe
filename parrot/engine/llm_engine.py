# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, AsyncGenerator
import asyncio

from parrot.constants import ENGINE_LOOP_INTERVAL, ENGINE_HEARTBEAT_INTERVAL
from parrot.protocol.layer_apis import register_engine
from parrot.utils import get_logger, set_random_seed, create_task_in_loop

from .config import EngineConfig


logger = get_logger("LLMEngine")


class LLMEngine:
    """Base class for all LLM engines. It provides a minimal interface for
    LLM engines."""

    def __init__(self, engine_config: Dict, connect_to_os: bool = True):
        # Set global random seed
        set_random_seed(engine_config["random_seed"])

        self.connect_to_os = connect_to_os
        if self.connect_to_os:
            assert (
                "os" in engine_config
            ), "If connect_to_os is True, os config must be provided."
            os_config = engine_config["os"]

            self.os_http_address = f"http://{os_config['host']}:{os_config['port']}"
        engine_config.pop("os")

    def _register_engine(self, engine_config: EngineConfig):
        """Register engine to OS."""

        if self.connect_to_os:
            resp = register_engine(
                http_addr=self.os_http_address,
                engine_config=engine_config,
            )
            self.engine_id = resp.engine_id
        else:
            self.engine_id = 0

    async def fill(self, payload: Dict) -> Dict:
        """Fill API.

        Args:
            payload: Dict[str, Any]. The payload of the fill API.

        Returns:
            Dict. The response of the fill API.
        """
        raise NotImplementedError

    async def generate(self, payload: Dict) -> Dict:
        """Generate API.

        Args:
            payload: Dict[str, Any]. The payload of the generate API.

        Returns:
            Dict. The response of the generate API.
        """
        raise NotImplementedError

    def generate_stream(self, payload: Dict) -> AsyncGenerator:
        """Generate stream API.

        Args:
            payload: Dict[str, Any]. The payload of the generate stream API.

        Returns:
            The generator of the generate stream API.
        """
        raise NotImplementedError

    def free_context(self, payload: Dict) -> Dict:
        """Free context API.

        Args:
            payload: Dict[str, Any]. The payload of the free context API.

        Returns:
            Dict. The response of the free context API.
        """
        raise NotImplementedError

    async def heartbeat(self):
        """Heartbeat sent to OS.

        Return: num_cached_tokens, cached_tokens_size. num_running_jobs."""
        raise NotImplementedError

    async def engine_iter(self):
        """The function executed in the every iteration of the engine loop."""
        raise NotImplementedError

    async def _heartbeat_loop(self):
        """Loop for heartbeat. It is registered in the same event loop with engine loop."""
        while True:
            await self.heartbeat()  # Send heartbeat to OS
            await asyncio.sleep(ENGINE_HEARTBEAT_INTERVAL)

    async def engine_loop(self):
        """Engine loop, execute jobs token by token.

        For some types of engines, e.g. OpenAI engine, the engine loop is empty loop.
        """

        # Create a task for heartbeat.
        create_task_in_loop(self._heartbeat_loop())

        while True:
            await asyncio.sleep(ENGINE_LOOP_INTERVAL)
            await self.engine_iter()
