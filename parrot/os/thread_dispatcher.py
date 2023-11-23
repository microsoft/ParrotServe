# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, Union, List
from enum import Enum
from queue import Queue
from dataclasses import dataclass

from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError
from .process.thread import Thread
from .engine import ExecutionEngine

logger = get_logger("ThreadDispatcher")


@dataclass
class DispatcherConfig:
    dag_aware: bool = False
    app_fifo: bool = False
    max_queue_size: int = 1024


class ThreadDispatcher:
    """ThreadDispatcher, or called ThreadScheduler, is responsible for dispatching threads
    to different backend engines.

    It is shared between different processes, so that it can has the global view of all threads
    from different processes. For exmaple, threads from different processes with the same prefix
    can be scheduled to the same engine.
    """

    def __init__(
        self,
        config: DispatcherConfig,
        engines: Dict[int, ExecutionEngine],
        flush_engine_callback=None,
    ):
        self.config = config
        self.engines = engines
        self.flush_engine_callback = flush_engine_callback
        self.thread_queue = Queue(self.config.max_queue_size)

        # Private states used in dispatching.
        self._engines_remain_locs: Dict[int, int] = {}

    def _get_engine_list(self, thread: Thread) -> List[ExecutionEngine]:
        engines_list = list(self.engines.values())
        models = thread.call.func.metadata.models
        max_jobs_num = thread.max_jobs_num

        def check_engine_available(engine: ExecutionEngine):
            # If models is empty, it means the function can be executed on any model.
            if models == [] or engine.config.model in models:
                return True

            # Check whether the engine fulfills the thread's max_jobs_num requirement.
            # This condition is only used in DAG-aware policy.
            if self.config.dag_aware and max_jobs_num < engine.jobs_num:
                return True

            # Check whether the engine has enough remain locs.
            return engine.remain_job_locs > 0

        # Get the available engines.
        return [engine for engine in engines_list if check_engine_available(engine)]

    def _dispatch_one(self, thread: Thread) -> bool:
        """Return if the thread is dispatched."""

        # Get the available engines.
        engines_list = self._get_engine_list(thread)

        # No available engine.
        if len(engines_list) == 0:
            return False

        # Get the best candidate engine.
        if self.config.dag_aware:
            # DAG Aware policy: select the engine with the most remain locs first,
            # preventing threads with a relaxed max_jobs_num requirement from
            # occupying the engine with a smaller remain locs.
            best_candidate_id = -1
            for engine_id, remain_locs in self._engines_remain_locs.items():
                if (
                    best_candidate_id == -1
                    or remain_locs < self._engines_remain_locs[best_candidate_id]
                ):
                    best_candidate_id = engine_id
        else:
            # Default policy: dispatch to the engine with the most remain locs.
            best_candidate_id = -1
            for engine_id, remain_locs in self._engines_remain_locs.items():
                if (
                    best_candidate_id == -1
                    or remain_locs > self._engines_remain_locs[best_candidate_id]
                ):
                    best_candidate_id = engine_id
            best_candidate = self.engines[best_candidate_id]

        self._engines_remain_locs[best_candidate.engine_id] -= 1
        best_candidate.accept_thread(thread)

        logger.info(f"Thread {thread.tid} dispatched to engine {best_candidate.name}.")

        return True

    # ---------- Public Methods ----------

    def push_thread(self, thread: Thread):
        """Push a thread to the thread queue."""

        if self.thread_queue.qsize() >= self.config.max_queue_size:
            raise ParrotOSUserError(
                RuntimeError(
                    f"Thread queue is full. Current size: {len(self.thread_queue)}. "
                    f"Hence the incoming thread (tid={thread.tid}) is rejected."
                )
            )

        self.thread_queue.put_nowait(thread)  # Append from right

    def dispatch(self) -> List[Thread]:
        """Dispatch all the (available) threads in the order of the queue."""

        # No thread to dispatch.
        if self.thread_queue.empty():
            return []

        dispatched_threads: List[Thread] = []

        # Flush engines.
        # To make sure the engine is alive, we need to ping it first and sweep the dead engines.
        # And ping the engines can also update the engine status.
        if self.flush_engine_callback is not None:
            self.flush_engine_callback(list(self.engines.values()))
        dead_keys = [key for key, engine in self.engines.items() if engine.dead]
        for key in dead_keys:
            self.engines.pop(key)

        # Maintain the remain locs
        self._engines_remain_locs = {
            engine_id: engine.remain_job_locs
            for engine_id, engine in self.engines.items()
        }

        # Dispatch all possible threads.
        new_thread_queue = Queue(self.config.max_queue_size)
        while not self.thread_queue.empty():
            thread: Thread = self.thread_queue.get()
            if not self._dispatch_one(thread):
                # If the process is not alive, discard the thread directly.
                if thread.process.live:
                    new_thread_queue.put_nowait(thread)
            else:
                # TODO(chaofan): App FIFO
                dispatched_threads.append(thread)
        self.thread_queue = new_thread_queue

        # Display the dispatch results.
        logger.debug(
            f"Dispatched {len(dispatched_threads)} threads. Results: \n"
            + "\n".join(
                [
                    f"  {thread.tid} -> engine: id={thread.engine.engine_id}, name={thread.engine.name}, "
                    for thread in dispatched_threads
                ]
            )
        )

        return dispatched_threads
