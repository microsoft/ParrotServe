from typing import Dict

from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError

from .process.thread import Thread
from .process.interpret_type import InterpretType
from .process.executor import Executor
from .engine import ExecutionEngine

logger = get_logger("ThreadDispatcher")


class ThreadDispatcher:
    """ThreadDispatcher, or called ThreadScheduler, is responsible for dispatching threads
    to different backend engines.

    It is shared between different processes, so that it can has the global view of all threads
    from different processes. For exmaple, threads from different processes with the same prefix
    can be scheduled to the same engine.
    """

    def __init__(self, engines: Dict[int, ExecutionEngine], flush_engine_callback=None):
        self.engines = engines
        self.flush_engine_callback = flush_engine_callback

    def dispatch(self, thread: Thread):
        """Dispatch a thread to some backend engine."""

        engines_list = list(self.engines.values())
        models = thread.call.func.metadata.models

        def check_model(engine: ExecutionEngine):
            # If models is empty, it means the function can be executed on any model.
            if models == []:
                return True
            return engine.config.model_name in models

        # Get the available engines.
        # To make sure the engine is alive, we need to ping it first and sweep the dead engines.
        available_engines = [engine for engine in engines_list if check_model(engine)]
        if self.flush_engine_callback is not None:
            self.flush_engine_callback(available_engines)
        available_engines = [engine for engine in available_engines if not engine.dead]

        # TODO: If no available engines, we report an error for now.
        if len(available_engines) == 0:
            raise ParrotOSUserError(
                RuntimeError("No live engine available. Thread dispatch failed.")
            )
        else:
            names = " ,".join([engine.name for engine in available_engines])
            logger.debug(
                f"Start dispatching. Total available engines ({len(available_engines)}): {names}."
            )

        thread.engine = available_engines[0]  # dispatched_engine

        logger.info(f"Thread {thread.tid} dispatched to engine {thread.engine.name}.")
