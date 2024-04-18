# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional, List, Set
from dataclasses import dataclass

from parrot.exceptions import ParrotCoreUserError
from parrot.utils import get_logger, RecyclePool

from parrot.serve.graph import RequestChain
from parrot.serve.backend_repr import ExecutionEngine
from parrot.serve.backend_repr.model import get_model_type, ModelType

from ..engine_manager import EngineManager
from ..context_manager import ServeCoreContextManager
from .completion_task import CompletionTask, TaskStatus


logger = get_logger("GlobalScheduler")


@dataclass
class GlobalSchedulerConfig:
    app_fifo: bool = False
    graph_group: bool = False
    ctx_group: bool = False
    ctx_aware: bool = False
    max_queue_size: int = 1024


class GlobalScheduler:
    """GlobalScheduler (GS) solves the task scheduling problem in the global scope."""

    def __init__(
        self,
        config: GlobalSchedulerConfig,
        engine_mgr: EngineManager,
        context_mgr: ServeCoreContextManager,
    ):
        # ---------- Basic ----------
        self.config = config
        self.engine_mgr = engine_mgr
        self.context_mgr = context_mgr

        # ---------- Task Queue ----------
        self.task_queue: List[CompletionTask] = []

    def _get_engine_list(
        self,
        tasks: List[CompletionTask],
        tasks_num_upperbound: int,
    ) -> List[ExecutionEngine]:
        engine_list = self.engine_mgr.get_live_engines()

        # NOTE(chaofan): Suppose all tasks noted the same "models" arg.
        models = tasks[0].chain.metadata.models
        model_type_str = tasks[0].chain.metadata.model_type
        model_type = get_model_type(model_type_str)
        # TODO(chaofan): Throughput/latency criteria

        def check_engine_available(engine: ExecutionEngine):
            # Check whether the mode type matches
            if model_type != engine.model.model_type:
                return False

            # Check whether the model matches
            if len(models) > 0 and engine.model_name not in models:
                return False

            # Check whether it violates the tasks_num_upperbound of the tasks.
            # NOTE(chaofan): For TaskGroup (i.e. tasks passed to this function),
            # the whole group is considered as a single task.
            if 1 + engine.get_num_tasks() > tasks_num_upperbound:
                return False

            # Check whether it violates the tasks_num_upperbound of the engine.
            if len(tasks) + engine.get_num_tasks() > engine.get_tasks_num_upperbound():
                return False

            # Check whether the engine has enough task capacity.
            if len(tasks) > engine.get_remain_tasks_capacity():
                return False
            
            if model_type == ModelType.TOKEN_ID:
                total_tokens_num = 0
                for task in tasks:
                    total_tokens_num += task.get_token_nums(engine.model.tokenizer_name)

                # Check whether the engine has enough token capacity.
                if total_tokens_num > engine.get_remain_tokens_capacity():
                    return False

            return True

        return [engine for engine in engine_list if check_engine_available(engine)]

    def _find_engine(self, tasks: List[CompletionTask]) -> None:
        """Find the best engine for a group of tasks."""

        tasks_num_upperbound = 999999999
        for task in tasks:
            tasks_num_upperbound = min(
                tasks_num_upperbound, task.schedule_annotation.tasks_num_upperbound
            )

        # Get the engine list
        engine_list = self._get_engine_list(tasks, tasks_num_upperbound)

        if len(engine_list) == 0:
            return

        # if len(engine_list) == 0:
        #     if len(tasks) == 1:
        #         return
        #     else:
        #         # Split the group
        #         for task in tasks:
        #             self._find_engine([task])
        #         return

        # Get the engines with Context
        # We use the first task's context to find the engines with the same context
        if self.config.ctx_aware:
            engine_ids_with_prefixes = self.context_mgr.query_prefixes_in_engines(
                tasks[0]
            )
            # print(engine_ids_with_prefixes)

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
                    engine.get_tasks_num_upperbound()
                    < best_engine.get_tasks_num_upperbound()
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
            task.schedule_to(best_engine)

    # ---------- Public Methods ----------

    def submit_task(self, task: CompletionTask) -> None:
        """Submit a task to the scheduler's queue."""

        if len(self.task_queue) >= self.config.max_queue_size:
            raise ParrotCoreUserError(
                RuntimeError(
                    f"Task queue is full. Current size: {len(self.task_queue)}. "
                    f"Hence the incoming task is rejected."
                )
            )

        logger.debug(
            f"Session(session_id={task.chain.session_id}) submit Task(task_id={task.task_id})"
            " to GlobalScheduler."
        )

        self.task_queue.append(task)
        task.status = TaskStatus.INQUEUE
        return

    def schedule(self) -> None:
        """Try to schedule all tasks in scheduler's queue."""

        if self.config.app_fifo:
            # Sort the tasks by the order of depth
            self.task_queue.sort(key=lambda x: x.chain.depth)

        # NOTE(chaofan): The tasks are sorted by priority, by default.
        for i, task in enumerate(self.task_queue):
            if task.is_scheduled:
                continue

            # Group tasks in rest queue
            cur_group: List[CompletionTask] = [task]
            chain_groups = set(task.chain.chain_groups)

            # Only allow one type of grouping at a time
            graph_group_enabled = self.config.graph_group
            ctx_group_enabled = self.config.ctx_group

            if graph_group_enabled or ctx_group_enabled:
                for j in range(i + 1, len(self.task_queue)):
                    task_j = self.task_queue[j]

                    # TODO(chaofan): Models match check
                    models_i = task.chain.metadata.models
                    models_j = task_j.chain.metadata.models

                    # TODO(chaofan): Criteria match check. Only group tasks with the same criteria.

                    # Graph group check
                    if graph_group_enabled:
                        chain_groups_j = set(task_j.chain.chain_groups)
                        common_groups = chain_groups.intersection(chain_groups_j)
                        if len(common_groups) > 0:
                            cur_group.append(task_j)
                            chain_groups = common_groups
                            ctx_group_enabled = False  # Use graph group this round

                    # Context group check
                    if ctx_group_enabled:
                        if task.chain.first_node.sv == task_j.chain.first_node.sv:
                            cur_group.append(task_j)
                            graph_group_enabled = False  # Use context group this round

            # Try to find engines for the group
            self._find_engine(cur_group)

        # Update the task queue
        prev_task_queue = self.task_queue
        scheduled_task = [task for task in prev_task_queue if task.is_scheduled]
        self.task_queue = [task for task in prev_task_queue if not task.is_scheduled]

        # Display the scheduled results.
        # NOTE(chaofan): Only display >0 case to reduce the log size.
        if len(scheduled_task) > 0:
            logger.debug(
                f"Scheduled {len(scheduled_task)} tasks. Results: \n"
                + "\n".join(
                    [
                        f"  Task {task.task_id} -> engine: id={task.engine.engine_id}, name={task.engine.name}, "
                        f"num_tasks={task.engine.get_num_tasks()}, "
                        f"remain_tasks_capacity={task.engine.get_remain_tasks_capacity()}, "
                        f"remain_tokens_capacity={task.engine.get_remain_tokens_capacity()}, "
                        f"tasks_num_upperbound={task.engine.get_tasks_num_upperbound()}, "
                        f"tokens_num={task.engine.get_tokens_num()}, "
                        for task in scheduled_task
                    ]
                )
            )

        return
