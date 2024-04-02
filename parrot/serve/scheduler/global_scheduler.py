# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, Union, List, Callable, Optional, Set
from dataclasses import dataclass

from parrot.exceptions import ParrotOSUserError
from parrot.utils import get_logger, RecyclePool

from parrot.serve.graph import CompletionChain, PlaceholderGen
from parrot.serve.backend_repr import ExecutionEngine

from ..engine_manager import EngineManager
from ..context_manager import ServeCoreContextManager
from .completion_task import CompletionTask


logger = get_logger("GlobalScheduler")


@dataclass
class GlobalSchedulerConfig:
    app_fifo: bool = False
    graph_group: bool = False
    ctx_group: bool = False
    ctx_aware: bool = False
    max_queue_size: int = 1024


class GlobalScheduler:
    """GlobalScheduler(GS) solves the task scheduling problem in the global scope."""

    def __init__(
        self,
        config: GlobalSchedulerConfig,
        engine_mgr: EngineManager,
        context_mgr: ServeCoreContextManager,
        ping_engine_method: Optional[Callable] = None,
    ):
        # ---------- Basic ----------
        self.config = config
        self.engine_mgr = engine_mgr
        self.context_mgr = context_mgr
        self.ping_engine_method = ping_engine_method

        # ---------- Task ----------
        # task_id -> task
        self.tasks: Dict[int, CompletionTask] = {}

        self.task_id_pool = RecyclePool("TaskIDPool")
        self.task_queue: List[CompletionTask] = []

    def _get_engine_list(
        self,
        tasks: List[CompletionTask],
        num_tasks_upperbound: int,
    ) -> List[ExecutionEngine]:
        engine_list = self.engine_mgr.get_live_engines()
        # NOTE(chaofan): Suppose all tasks noted the same "models" arg.
        models = tasks[0].chain.request_chain.metadata.models
        # TODO(chaofan): Throughput/latency criteria

        def check_engine_available(engine: ExecutionEngine):
            total_tokens_num = 0
            for task in tasks:
                total_tokens_num += task.get_token_nums(engine.model.tokenizer_name)

            # Check whether it violates the num_tasks_upperbound of the tasks.
            # NOTE(chaofan): For TaskGroup (i.e. tasks passed to this function),
            # the whole group is considered as a single task.
            if 1 + engine.get_num_tasks() > num_tasks_upperbound:
                return False

            # Check whether it violates the num_tasks_upperbound of the engine.
            if len(tasks) + engine.get_num_tasks() > engine.get_num_tasks_upperbound():
                return False

            # Check whether the engine has enough token capacity.
            if total_tokens_num > engine.get_remain_tokens_capacity():
                return False

            # Check whether the engine has enough task capacity.
            if len(tasks) > engine.get_remain_tasks_capacity():
                return False

            return True

        return [engine for engine in engine_list if check_engine_available(engine)]

    def _find_engine(self, tasks: List[CompletionTask]) -> None:
        """Find the best engine for a group of tasks."""

        num_tasks_upperbound = 999999999
        for task in tasks:
            num_tasks_upperbound = min(num_tasks_upperbound, task.num_tasks_upperbound)

        # Get the engine list
        engine_list = self._get_engine_list(tasks, num_tasks_upperbound)

        if len(engine_list) == 0:
            if len(tasks) == 1:
                return
            else:
                # Split the group
                for task in tasks:
                    self._find_engine([task])
                return

        # Get the engines with Context
        # We use the first task's context to find the engines with the same context
        if self.config.ctx_aware:
            engine_ids_with_prefixes = self.context_mgr.query_prefixes_in_engines(
                tasks[0]
            )

        best_engine = None
        for engine in engine_list:
            if best_engine is None:
                best_engine = engine
            elif (
                self.config.ctx_aware
                and engine.engine_id in engine_ids_with_prefixes
                and best_engine.engine_id not in engine_ids_with_prefixes
            ):
                # Context-aware engine is preferred
                best_engine = engine
            else:
                # Select the best engine (minimizing the negative impacts, i.e. minimizing the decreasing of upperbound)
                # If the upperbound is not affected, select the engine with the most capacity.
                if (
                    engine.get_num_tasks_upperbound()
                    < best_engine.get_num_tasks_upperbound()
                ):
                    best_engine = engine
                elif (
                    engine.get_remain_tokens_capacity()
                    < best_engine.get_remain_tokens_capacity()
                ):
                    best_engine = engine

        # Dispatch the tasks to the engine
        assert best_engine is not None
        for task in tasks:
            task.engine = best_engine
            task.scheduled.set()
            best_engine.update_servelayer_runtime_info(
                task_id=task.task_id,
                tokens_num=task.get_token_nums(best_engine.model.tokenizer_name),
                num_tasks_upperbound=task.num_tasks_upperbound,
            )

    # ---------- Public Methods ----------

    def submit_completion(self, chain: CompletionChain) -> CompletionTask:
        """Submit a completion chain to the scheduler.

        Returns:
            A CompletionTask object representing the task.
        """

        if len(self.task_queue) >= self.config.max_queue_size:
            raise ParrotOSUserError(
                RuntimeError(
                    f"Task queue is full. Current size: {len(self.task_queue)}. "
                    f"Hence the incoming task is rejected."
                )
            )

        task_id = self.task_id_pool.allocate()
        task = CompletionTask(task_id=task_id, chain=chain)
        self.task_queue.append(task)
        self.tasks[task_id] = task
        return task

    def schedule(self) -> None:
        """Try to schedule all tasks in scheduler's queue."""

        # NOTE(chaofan): The tasks are sorted by priority, by default.
        checked_tasks: Set[int] = set()
        for i, task in enumerate(self.task_queue):
            if task.task_id in checked_tasks:
                continue

            # Group tasks in rest queue
            cur_group: List[CompletionTask] = [task]
            checked_tasks.add(task.task_id)
            producers: Set[PlaceholderGen] = set(task.chain.get_producers())

            # Only allow one type of grouping at a time
            graph_group_enabled = self.config.graph_group
            ctx_group_enabled = self.config.ctx_group

            if graph_group_enabled or ctx_group_enabled:
                for j in range(i + 1, len(self.task_queue)):
                    task_j = self.task_queue[j]

                    # TODO(chaofan): Models match check
                    models_i = task.chain.request_chain.metadata.models
                    models_j = task_j.chain.request_chain.metadata.models

                    # TODO(chaofan): Criteria match check. Only group tasks with the same criteria.

                    # Graph group check
                    if graph_group_enabled:
                        producers_j: Set[PlaceholderGen] = set(
                            task_j.chain.get_producers()
                        )
                        if producers.intersection(producers_j):
                            cur_group.append(task_j)
                            checked_tasks.add(task_j.task_id)
                            ctx_group_enabled = False

                    # Context group check
                    if ctx_group_enabled:
                        if task.chain.first_node.sv_id == task_j.chain.first_node.sv_id:
                            cur_group.append(task_j)
                            checked_tasks.add(task_j.task_id)
                            graph_group_enabled = False

            # Try to find engines for the group
            self._find_engine(cur_group)

        # Update the task queue
        prev_task_queue = self.task_queue
        scheduled_task = [
            task for task in prev_task_queue if task.task_id in checked_tasks
        ]
        self.task_queue = [
            task for task in prev_task_queue if task.task_id not in checked_tasks
        ]

        # Display the scheduled results.
        # NOTE(chaofan): Only display >0 case to reduce the log size.
        if len(checked_tasks) > 0:
            logger.debug(
                f"Scheduled {len(checked_tasks)} tasks. Results: \n"
                + "\n".join(
                    [
                        f"  Task {task.task_id} -> engine: id={task.engine.engine_id}, name={task.engine.name}, "
                        f"num_tasks={task.engine.get_num_tasks()}, "
                        f"remain_tasks_capacity={task.engine.get_remain_tasks_capacity()}, "
                        f"remain_tokens_capacity={task.engine.get_remain_tokens_capacity()}, "
                        f"num_tasks_upperbound={task.engine.get_num_tasks_upperbound()}, "
                        f"tokens_num={task.engine.get_tokens_num()}, "
                        for task in scheduled_task
                    ]
                )
            )

        return
