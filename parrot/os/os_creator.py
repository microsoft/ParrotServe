# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import json
from typing import Dict

from parrot.utils import get_logger

from .pcore import PCore

from parrot.os.config import OSConfig
from parrot.exceptions import ParrotOSInternalError

logger = get_logger("OS Creator")


def create_os(
    os_config_path: str,
    release_mode: bool = False,
    override_args: Dict = {},
) -> PCore:
    """Create the PCore.

    Args:
        os_config_path: str. The path to the OS config file.
        release_mode: bool. Whether to run in release mode.
        override_args: Dict. The override arguments.

    Returns:
        PCore. The created Parrot OS Core.
    """

    with open(os_config_path) as f:
        os_config = dict(json.load(f))

    os_config.update(override_args)

    if not OSConfig.verify_config(os_config):
        raise ParrotOSInternalError(f"Invalid OS config: {os_config}")

    return PCore(os_config)
