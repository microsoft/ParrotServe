# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict
from parrot.utils import get_logger, RecyclePool

from parrot.serve.graph import CompletionChain, PerformanceCriteria

from .completion_task import CompletionTask
from .schedule_annotation import ScheduleAnnotation


logger = get_logger("TaskCreator")


class TaskCreator:
    """TaskCreator creates a CompletionTask object for the CompletionChain.

    It provides a primitive function in Parrot system: Performance deduction.

    The algorithm works as follows:
    1. For a given CompletionChain, the TaskCreator first checks the task_cache to see if the task has
        been created before.
    2. If it's not in the cache, the TaskCreator creates a new Task object for the chain by lowering
        the PerformanceCriteria to SchedulerAnnotation.
    3. Then the TaskCreator will do a performance deduction in the Graph. It starts from the Gen node of
        this task, traverses backward to its predecessors, propagates the performance deduction result and
        also converts them into Task objects (stored in task_cache).
    """

    def __init__(self) -> None:
        # gen_sv_id -> task object
        self.task_cache: Dict[str, CompletionTask] = {}

        self.task_id_pool = RecyclePool("TaskIDPool")

    def _lower_criteria(self, criteria: PerformanceCriteria) -> ScheduleAnnotation:
        if criteria == PerformanceCriteria.LATENCY:
            return ScheduleAnnotation(
                tasks_num_upperbound=4,
                tokens_num_upperbound=4096,
            )
        elif criteria == PerformanceCriteria.THROUGHPUT:
            return ScheduleAnnotation(
                tasks_num_upperbound=99999,
                tokens_num_upperbound=9999999999999,
            )
        else:
            raise NotImplementedError(
                f"PerformanceCriteria {criteria} is not supported."
            )

    def _back_propagate_criteria(
        self, criteria: PerformanceCriteria
    ) -> PerformanceCriteria:
        if criteria == PerformanceCriteria.LATENCY:
            return PerformanceCriteria.LATENCY
        elif criteria == PerformanceCriteria.THROUGHPUT:
            return PerformanceCriteria.THROUGHPUT
        else:
            raise NotImplementedError(
                f"PerformanceCriteria {criteria} is not supported."
            )

    def create_task(
        self, completion_chain: CompletionChain, criteria: PerformanceCriteria
    ) -> CompletionTask:
        """Create a Task object for the CompletionChain.

        Args:
            completion_chain: CompletionChain.
            criteria: PerformanceCriteria.

        Returns:
            CompletionTask. The Task object created for the CompletionChain.
        """

        if completion_chain.gen_node.sv_id in self.task_cache:
            return self.task_cache[completion_chain.gen_node.sv_id]

        # Create a new Task
        task_id = self.task_id_pool.allocate()
        schedule_annotation = self._lower_criteria(criteria)
        task = CompletionTask(
            task_id=task_id,
            chain=completion_chain,
            schedule_annotation=schedule_annotation,
        )
        self.task_cache[completion_chain.gen_node.sv_id] = task

        # Traverse back and do performance deduction
        next_criteria = self._back_propagate_criteria(criteria)
        for node in completion_chain.iter_fill():
            if node.sv.producer is not None:
                self.create_task(node.sv.producer.completion_chain, next_criteria)

        return task

    def free_task(self, task: CompletionTask) -> None:
        """Free the CompletionTask.

        Args:
            task: CompletionTask. The task to be freed.
        """

        self.task_id_pool.free(task.task_id)
        self.task_cache.pop(task.chain.gen_node.sv_id)
        return
