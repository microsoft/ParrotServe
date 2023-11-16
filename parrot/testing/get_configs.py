# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Get config files by relative path to the package root path."""

import parrot


def get_engine_config_path(engine_type: str, config_file_name: str) -> str:
    # The config path is relative to the package path.
    # We temporarily use this way to load the config.
    package_path = parrot.__path__[0]
    return f"{package_path}/../configs/engine/{engine_type}/{config_file_name}"


def get_os_config_path(config_file_name: str) -> str:
    package_path = parrot.__path__[0]
    return package_path + "/../configs/os/" + config_file_name
