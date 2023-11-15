# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import logging
import asyncio
from asyncio import Task
import traceback
import sys
from typing import Optional, List, Coroutine


def get_logger(log_name: str, log_level: int = logging.DEBUG):
    logger = logging.getLogger(log_name)

    if not logger.handlers:
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


class RecyclePool:
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.free_ids: List[int] = list(range(pool_size))

    def allocate(self) -> int:
        """Fetch an id."""

        if len(self.free_ids) == 0:
            raise ValueError("No free id in the pool.")
        allocated_id = self.free_ids.pop()
        return allocated_id

    def free(self, id: int) -> int:
        """Free an id."""

        if id in self.free_ids:
            raise ValueError("The id is already free.")

        self.free_ids.append(id)


def _task_error_callback_fail_fast(task):
    if not task.cancelled() and task.exception() is not None:
        e = task.exception()
        print("--- QUIT THE WHOLE SYSTEM BECAUSE ERROR HAPPENS! (Fail Fast Mode) ---")
        traceback.print_exception(None, e, e.__traceback__)
        sys.exit(1)  # Quit everything if there is an error


def create_task_in_loop(
    coro: Coroutine,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    fail_fast: bool = True,
):
    if loop is None:
        loop = asyncio.get_running_loop()
    # asyncio.run_coroutine_threadsafe(coro, loop)
    # asyncio.create_task(coro)
    task = loop.create_task(coro)
    if fail_fast:
        task.add_done_callback(_task_error_callback_fail_fast)
    return task


def set_random_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
