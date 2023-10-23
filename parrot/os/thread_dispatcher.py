from typing import Dict

from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError

from .process.thread import Thread
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
        if self.flush_engine_callback is not None:
            self.flush_engine_callback()
        live_engines_list = [engine for engine in engines_list if not engine.dead]

        # TODO: Implement the dispatching strategy.
        if len(live_engines_list) == 0:
            raise ParrotOSUserError(
                RuntimeError("No live engine available. Thread dispatch failed.")
            )

        thread.engine = live_engines_list[0]

        logger.info(f"Thread {thread.tid} dispatched to engine {thread.engine.name}.")
