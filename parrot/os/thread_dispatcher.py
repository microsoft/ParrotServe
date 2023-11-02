from typing import Dict

from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError
from parrot.constants import LATENCY_AWARE_BS_THRESHOLD

from .process.placeholder import Placeholder
from .process.thread import Thread
from .engine import ExecutionEngine

logger = get_logger("ThreadDispatcher")


class Candidate:
    def __init__(self, engine: ExecutionEngine):
        self.engine = engine
        self.score = 0


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

        # Then we start dispatching it to the most suitable engine.
        candidates = [Candidate(engine) for engine in available_engines]

        # If expect_batch_size is large, we should schedule it to a non-latency-aware engine.
        expect_batch_size = 0
        for name, value in thread.call.bindings.items():
            if thread.call.func.params_map[name].is_output:
                assert isinstance(value, Placeholder)
                cur_batch_size = sum([node.in_degree for node in value.out_nodes])
                expect_batch_size = max(expect_batch_size, cur_batch_size)
        logger.debug(
            f"Call {thread.call.func.name} expect batch size: {expect_batch_size}."
        )

        if expect_batch_size >= LATENCY_AWARE_BS_THRESHOLD:
            for candidate in candidates:
                candidate.score -= 999

        # According to the remain batch size, assign a score to each engine.
        for candidate in candidates:
            candidate.score += candidate.engine.remain_batch_size

        # Schedule to the engine with the highest score.
        best_candidate = candidates[0]
        for candidate in candidates[1:]:
            if candidate.score > best_candidate.score:
                best_candidate = candidate

        thread.engine = best_candidate.engine
        thread.engine.num_threads += 1

        logger.info(f"Thread {thread.tid} dispatched to engine {thread.engine.name}.")
