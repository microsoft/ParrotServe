# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import logging
import asyncio
from asyncio import Task
import traceback
import sys
import os
from typing import Optional, List, Coroutine


### Logging ###

log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log_file_path = None
loggers: List[logging.Logger] = []


def _set_log_handler(logger: logging.Logger, log_level: int):
    global log_file_path
    logger.setLevel(log_level)
    if log_file_path is not None:
        handler = logging.FileHandler(log_file_path)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)


def _flush_handlers():
    global loggers

    for logger in loggers:
        if logger.hasHandlers():
            log_level = logger.handlers[0].level
            logger.removeHandler(logger.handlers[0])
            _set_log_handler(logger, log_level)


def set_log_output_file(log_file_dir_path: str, log_file_name: str):
    """Set the file logger."""

    global log_formatter
    global log_file_path

    makedir_flag = False
    if not os.path.exists(log_file_dir_path):
        os.makedirs(log_file_dir_path)
        makedir_flag = True

    log_file_path = os.path.join(log_file_dir_path, log_file_name)

    # Redirect stdout and stderr to the log file
    fp = open(log_file_path, "w")
    sys.stdout = fp
    sys.stderr = fp

    if makedir_flag:
        print(
            "The log directory does not exist. Create log file directory: ",
            log_file_dir_path,
        )

    _flush_handlers()


def get_logger(log_name: str, log_level: int = logging.DEBUG):
    """Get a logger with the given name and the level."""

    global log_formatter
    global log_handler

    logger = logging.getLogger(log_name)
    if logger not in loggers:
        loggers.append(logger)

    if not logger.hasHandlers():
        _set_log_handler(logger, log_level)

    return logger


###


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
