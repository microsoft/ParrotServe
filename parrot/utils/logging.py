# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import logging
import os
from typing import List


log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log_file_path = None
loggers: List[logging.Logger] = []


def _set_log_handler(logger: logging.Logger, log_level: int):
    global log_file_path

    logger.setLevel(log_level)
    if log_file_path is not None:
        handler = logging.FileHandler(log_file_path, mode="a+", delay=False)
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

    if makedir_flag:
        print(
            "The log directory does not exist. Create log file directory: ",
            log_file_dir_path,
        )

    _flush_handlers()


def get_logger(log_name: str, log_level: int = logging.DEBUG):
    """Get a logger with the given name and the level."""

    global log_formatter

    logger = logging.getLogger(log_name)
    if logger not in loggers:
        loggers.append(logger)

    if not logger.hasHandlers():
        _set_log_handler(logger, log_level)

    return logger
