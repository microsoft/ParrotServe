# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, Union, List, Callable, Optional
from enum import Enum
from dataclasses import dataclass
from queue import Queue

from parrot.program.semantic_variable import Constant
from parrot.utils import get_logger, cprofile
from parrot.exceptions import ParrotOSUserError, parrot_assert

from .process.thread import Thread
from .engine import ExecutionEngine
from .memory.mem_space import MemorySpace

logger = get_logger("ThreadDispatcher")


@dataclass
class DispatcherConfig:
    dag_aware: bool = False
    app_fifo: bool = False
    ctx_aware: bool = False
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
        ping_engine_method: Optional[Callable] = None,
        memory_space: Optional[MemorySpace] = None,
    ):
        self.config = config
        self.engines = engines
        self.live_engines: Dict[int, ExecutionEngine] = {}
        self.ping_engine_method = ping_engine_method
        self.memory_space = memory_space

        self.threads: Dict[str, Thread] = {}  # uid -> thread
        self.thread_queue = Queue(maxsize=self.config.max_queue_size)

        self._flushed = False

    def _get_engine_list(self, thread: Thread) -> List[ExecutionEngine]:
        engines_list = list(self.live_engines.values())
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

            # Check whether the engine has enough token capacity.
            token_nums = engine.count_thread_token_nums(thread)
            if engine.tokens_num + token_nums > engine.config.tokens_capacity:
                return False

            # Check whether the engine has enough remain locs (threads num capacity).
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

        best_candidate = None

        if self.config.ctx_aware:
            parrot_assert(self.memory_space is not None, "Memory space is not set when Ctx Aware.")
            prefix = thread.call.func.prefix
            # TODO(chaofan): Prefix not always a constant. Fix it in the future.
            parrot_assert(isinstance(prefix, Constant), "Prefix is not a constant.")
            engine_ids_with_ctx = set(self.memory_space.get_engines_with_ctx(prefix.text))
            for engine in engines_list:
                if engine.engine_id in engine_ids_with_ctx:
                    # TODO(chaofan): In this policy, we select the first engine.
                    # It's OK for now, but we may need to improve it in the future.
                    best_candidate = engine
                    break
        if best_candidate is None:
            if self.config.dag_aware:
                # DAG Aware policy: select the engine with the least remain locs first,
                # preventing threads with a relaxed threads_capacity requirement from
                # occupying the engine with a smaller remain locs.
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
            f"Thread {thread.unique_id} dispatched to engine {best_candidate.name} (id={best_candidate.engine_id})."
        )

        return True

    def _dispatch(self) -> List[Thread]:
        # No thread to dispatch.
        if self.thread_queue.qsize() == 0:
            return []

        # Flush engines.
        # To make sure the engine is alive, we need to ping it first and sweep the dead engines.
        # And ping the engines can also update the engine status.
        if self.ping_engine_method is not None and not self._flushed:
            # NOTE(chaofan): It's slow. Fix in the future.
            # for _, engine in self.engines.items():
            #     self.ping_engine_method(engine)
            self._flushed = True

        self.live_engines = {
            engine_id: engine
            for engine_id, engine in self.engines.items()
            if not engine.dead
        }

        # with cprofile("dispatch"):
        dispatched_threads: List[Thread] = []

        new_thread_queue = []

        # Currently only two levels: 0 for normal, 1 for promoted.
        priority: Dict[str, int] = {}

        while self.thread_queue.qsize() > 0:
            thread: Thread = self.thread_queue.get()
            parrot_assert(
                thread.unique_id in self.threads,
                f"Thread not in the thread queue. {self.threads}",
            )

            if (
                len(dispatched_threads) > 0
                or not thread.ready_to_dispatch()
                or not self._dispatch_one(thread)
            ):
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
                            next_thread.unique_id in self.threads
                            and next_thread.ready_to_dispatch()
                        ):
                            logger.debug(
                                f"Thread (tid={next_thread.tid}) promoted by thread {thread.tid}. (pid={thread.process.pid})"
                            )
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
                        f"  Thread {thread.unique_id} -> engine: id={thread.engine.engine_id}, name={thread.engine.name}, "
                        f"num_threads={thread.engine.num_threads}, "
                        f"thread_capacity={thread.engine.config.threads_capacity}, "
                        f"tokens_capacity={thread.engine.config.tokens_capacity}, "
                        f"num_threads_upperbound={thread.engine.requests_num_upperbound}, "
                        f"tokens_num={thread.engine.tokens_num}, "
                        for thread in dispatched_threads
                    ]
                )
            )

        self._flushed = False

        return dispatched_threads
