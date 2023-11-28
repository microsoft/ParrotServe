# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, Union, List
from enum import Enum
from dataclasses import dataclass
from queue import Queue

from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError, parrot_assert
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
        ping_engine_method=None,
    ):
        self.config = config
        self.engines = engines
        self.ping_engine_method = ping_engine_method

        self.threads: Dict[str, Thread] = {}  # uid -> thread
        self.thread_queue = Queue(maxsize=self.config.max_queue_size)

    def _get_engine_list(self, thread: Thread) -> List[ExecutionEngine]:
        engines_list = list(self.engines.values())
        models = thread.call.func.metadata.models
        request_upperbound = thread.requests_num_upperbound

        def check_engine_available(engine: ExecutionEngine):
            # Check whether the model is supported by the engine.
            # If "models" field is empty, it means the function can be executed on any model.
            if len(models) > 0 and not engine.config.model in models:
                return False

            # Check whether the engine exceeds the uppderbound itself.
            # This condition is only used in DAG-aware policy.
            if (
                self.config.dag_aware
                and engine.requests_num_upperbound <= engine.num_threads
            ):
                return False

            # Check whether the engine fulfills the thread's num_threads requirement.
            # This condition is only used in DAG-aware policy.
            if self.config.dag_aware and request_upperbound <= engine.num_threads:
                return False

            # Check whether the engine has enough remain locs.
            return engine.remain_thread_locs > 0

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
            # DAG Aware policy: select the engine with the least remain locs first,
            # preventing threads with a relaxed max_threads_num requirement from
            # occupying the engine with a smaller remain locs.
            best_candidate = None
            for engine in engines_list:
                if (
                    best_candidate == None
                    or engine.remain_thread_locs < best_candidate.remain_thread_locs
                ):
                    best_candidate = engine
        else:
            # Default policy: dispatch to the engine with the most remain locs.
            best_candidate = None
            for engine in engines_list:
                if (
                    best_candidate == None
                    or engine.remain_thread_locs > best_candidate.remain_thread_locs
                ):
                    best_candidate = engine

        best_candidate.accept_thread(thread)

        logger.info(
            f"Thread {thread.tid} dispatched to engine {best_candidate.name} (id={best_candidate.engine_id})."
        )

        return True

    def _dispatch(self) -> List[Thread]:
        # No thread to dispatch.
        if self.thread_queue.qsize() == 0:
            return []

        dispatched_threads: List[Thread] = []

        # Flush engines.
        # To make sure the engine is alive, we need to ping it first and sweep the dead engines.
        # And ping the engines can also update the engine status.
        if self.ping_engine_method is not None:
            for _, engine in self.engines.items():
                self.ping_engine_method(engine)

        dead_keys = [key for key, engine in self.engines.items() if engine.dead]
        for key in dead_keys:
            self.engines.pop(key)

        # Dispatch all possible threads.
        new_thread_queue = []

        # Currently only two levels: 0 for normal, 1 for promoted.
        priority: Dict[str, int] = {}

        while self.thread_queue.qsize() > 0:
            thread: Thread = self.thread_queue.get()
            parrot_assert(
                thread.unique_id in self.threads,
                f"Thread not in the thread queue. {self.threads}",
            )

            if not thread.ready_to_dispatch() or not self._dispatch_one(thread):
                # If the process is not alive, discard the thread directly.
                if thread.process.live:
                    new_thread_queue.append(thread)
                    if thread.unique_id not in priority:
                        priority[thread.unique_id] = 0
            else:
                dispatched_threads.append(thread)
                self.threads.pop(thread.unique_id)

                # App FIFO: the thread will "pull" its successors to the top of the queue.
                if self.config.app_fifo:
                    next_threads = thread.get_next_threads()
                    for next_thread in next_threads:
                        if (
                            next_thread.ready_to_dispatch()
                            and next_thread.unique_id in self.threads
                        ):
                            priority[next_thread.unique_id] = 1

        # Reorder
        new_thread_queue.sort(
            key=lambda thread: priority[thread.unique_id], reverse=True
        )

        for thread in new_thread_queue:
            self.thread_queue.put_nowait(thread)

        return dispatched_threads

    # ---------- Public Methods ----------

    def push_thread(self, thread: Thread):
        """Push a thread to the thread queue."""

        if self.thread_queue.qsize() >= self.config.max_queue_size:
            raise ParrotOSUserError(
                RuntimeError(
                    f"Thread queue is full. Current size: {self.thread_queue.qsize()}. "
                    f"Hence the incoming thread (tid={thread.tid}) is rejected."
                )
            )

        self.thread_queue.put_nowait(thread)
        self.threads[thread.unique_id] = thread

    def dispatch(self) -> List[Thread]:
        """Dispatch all the threads in the queue."""

        newly_dispatched = self._dispatch()
        dispatched_threads = []
        dispatched_threads.extend(newly_dispatched)

        while len(newly_dispatched) > 0:
            newly_dispatched = self._dispatch()
            dispatched_threads.extend(newly_dispatched)

        # Display the dispatch results.
        # NOTE(chaofan): Only display >0 case to reduce the log size.
        if len(dispatched_threads) > 0:
            logger.debug(
                f"Dispatched {len(dispatched_threads)} threads. Results: \n"
                + "\n".join(
                    [
                        f"  tid={thread.tid} -> engine: id={thread.engine.engine_id}, name={thread.engine.name}, "
                        f"num_threads={thread.engine.num_threads}, num_threads_upperbound={thread.engine.requests_num_upperbound}, "
                        for thread in dispatched_threads
                    ]
                )
            )

        return dispatched_threads
