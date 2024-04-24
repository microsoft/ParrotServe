# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import traceback
import sys

from typing import Optional, List, Coroutine


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
