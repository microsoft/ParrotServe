# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

from parrot.utils import get_logger

from parrot.serve.graph import ComputeGraph

from ..global_scheduler import GlobalScheduler, ScheduleUnitTask
from ..tokenizer_wrapper import TokenizersWrapper


logger = get_logger("GraphPoller")


class GraphPoller:
    """GraphPoller in a session polls CompletionChain to GlobalScheduler in a background event loop."""

    def __init__(
        self,
        session_id: int,
        graph: ComputeGraph,
        scheduler: GlobalScheduler,
        tokenizers_wrapper: TokenizersWrapper,
    ):
        # ---------- Basic Info ----------
        self.session_id = session_id
        self.graph = graph

        # ---------- Global Components ----------
        self.scheduler = scheduler
        self.tokenizers_wrapper = tokenizers_wrapper

    async def loop(self):
        """Poll the graph and schedule the tasks to the global scheduler."""

        while True:
            # Poll the graph
            task = self.graph.poll()
            if task is None:
                await asyncio.sleep(0.1)
                continue

            # Schedule the task
            schedule_unit = ScheduleUnitTask(task_id=task.task_id, chain=task.chain)
            self.scheduler.schedule_task(schedule_unit)

            logger.debug(f"Task {task.task_id} is scheduled.")
