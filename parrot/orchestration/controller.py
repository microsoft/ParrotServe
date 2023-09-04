from typing import Dict, List
import logging
import time

from ..program.function import ParrotFunction
from .engine import ExecutionEngine
from ..utils import get_logger
from ..protocol import check_heartbeat


logger = get_logger("Global", logging.INFO)


class Controller:
    """Global controller."""

    def __init__(self):
        # ---------- Registry ----------
        self.engines_table: Dict[str, ExecutionEngine] = {}
        self.functions: List[ParrotFunction] = {}

    def register_engine(self, name: str, host: str, port: int):
        engine = ExecutionEngine(
            name=name,
            host=host,
            port=port,
        )

        try:
            resp = check_heartbeat(engine.http_address)
            assert resp.model_ready
        except BaseException as e:
            logger.error(f"Register engine error: {e}")
            return

        engine.cached_tokens = resp.cached_tokens
        engine.running_jobs = resp.running_jobs
        self.engines_table[name] = engine

        logger.info(
            f"Register execution engine: {engine.name} in {engine.http_address}"
        )

    def register_function(self, function: ParrotFunction):
        self.functions.append(function)
        logger.info(f"Register parrot function: {function.name}")

    def heartbeat_thread(self):
        # These configs are hardcoded.
        heartbeat_interval = 5  # (Unit: second)
        retry_times = 5

        while True:
            disconnect_engines: List[str] = []
            for engine in self.engines_table.values():
                resp = None
                for _ in range(retry_times):
                    try:
                        resp = check_heartbeat(engine.http_address)
                    except:
                        pass
                if resp is None:
                    disconnect_engines.append(engine.name)
                else:
                    engine.cached_tokens = resp.cached_tokens
                    engine.running_jobs = resp.running_jobs

            for engine_name in disconnect_engines:
                self.engines_table.pop(engine_name)

            time.sleep(heartbeat_interval)


# Singleton
parrot_global_ctrl = Controller()
