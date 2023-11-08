"""Engine Layer."""

import mlc_chat  # Avoid MLC error because "torch" is imported before "mlc_chat"


# This is a hack to override the __post_init__ method of NativeConfig
from .config import NativeConfig
from .native.native_config_post_init import __post_init__

NativeConfig.__post_init__ = __post_init__
# print("NativeConfig.__post_init__ is overriden.")
