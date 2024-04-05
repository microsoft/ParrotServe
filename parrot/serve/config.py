# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from dataclasses import dataclass
from typing import Dict

from parrot.constants import DEFAULT_SERVER_HOST, DEFAULT_OS_SERVER_PORT


@dataclass
class ServeCoreConfig:
    """Config for launching ServeCore."""

    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_OS_SERVER_PORT
    max_sessions_num: int = 2048
    max_engines_num: int = 2048
    session_life_span: int = 600

    @classmethod
    def verify_config(cls, config: Dict) -> bool:
        """Verify the ServeOS config.

        The ServeOS config should contain the following fields:
        - host: str
        - port: int
        - max_sessions_num: int
        - max_engines_num: int
        - session_life_span: int
        - global_scheduler: Dict (Global scheduler config)
        """

        if "global_scheduler" not in config:
            return False

        return True
