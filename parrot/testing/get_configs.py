# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Get config files by relative path to the package root path.

NOTE(chaofan): This part of functionality requires the original repo structure to be kept.

In particular, it will automatically search in the `configs` folder for the config file.
"""

import parrot


def get_sample_engine_config_path(config_file_name: str) -> str:
    # The config path is relative to the package path.
    # We temporarily use this way to load the config.
    package_path = parrot.__path__[0]
    return f"{package_path}/../sample_configs/engine/{config_file_name}"


def get_sample_os_config_path(config_file_name: str) -> str:
    package_path = parrot.__path__[0]
    return package_path + "/../sample_configs/os/" + config_file_name
