# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass
from typing import Dict

from parrot.constants import DEFAULT_SERVER_HOST, DEFAULT_OS_SERVER_PORT


@dataclass
class OSConfig:
    """Config for OS."""

    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_OS_SERVER_PORT
    max_proc_num: int = 2048
    max_engines_num: int = 2048

    @classmethod
    def verify_config(cls, config: Dict) -> bool:
        """Verify the OS config."""

        if "dispatcher" not in config:
            return False

        return True
