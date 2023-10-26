from typing import Dict, AsyncGenerator
import asyncio
import time

from parrot.constants import ENGINE_LOOP_INTERVAL, ENGINE_HEARTBEAT_INTERVAL
from parrot.protocol.layer_apis import register_engine
from parrot.utils import get_logger

from .config import EngineConfig


logger = get_logger("LLMEngine")


class LLMEngine:
    """Base class for all LLM engines. It provides a minimal interface for
    LLM engines."""

    def __init__(self, engine_config: Dict, connect_to_os: bool = True):
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

    async def generate(self, payload: Dict) -> Dict:
        """Generate API.

        Args:
            payload: Dict[str, Any]. The payload of the generate API.

        Returns:
            Dict. The response of the generate API.
        """

    def generate_stream(self, payload: Dict) -> AsyncGenerator:
        """Generate stream API.

        Args:
            payload: Dict[str, Any]. The payload of the generate stream API.

        Returns:
            The generator of the generate stream API.
        """

    def free_context(self, payload: Dict) -> Dict:
        """Free context API.

        Args:
            payload: Dict[str, Any]. The payload of the free context API.

        Returns:
            Dict. The response of the free context API.
        """

    async def heartbeat(self):
        """Heartbeat sent to OS.

        Return: num_cached_tokens, cached_tokens_size. num_running_jobs."""

    def engine_iter(self):
        """The function executed in the every iteration of the engine loop."""

    async def engine_loop(self):
        last_heartbeat_time = -1e14  # So that we can send heartbeat at the beginning

        while True:
            # Send heartbeat to OS
            cur_time = time.perf_counter_ns()
            if (cur_time - last_heartbeat_time) / 1e9 > ENGINE_HEARTBEAT_INTERVAL:
                await self.heartbeat()
                last_heartbeat_time = cur_time

            await asyncio.sleep(ENGINE_LOOP_INTERVAL)

            self.engine_iter()
