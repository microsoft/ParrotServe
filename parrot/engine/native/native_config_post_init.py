import torch
from .mem import MemLayout, ATTN_FUNC_LAYOUT_MAP
from .attn_func import ATTN_FUNC_MAP
from ..config import NativeConfig


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def __post_init__(native_config: NativeConfig):
    # Replace dtype and device
    native_config.dtype_str = native_config.dtype
    native_config.device_str = native_config.device
    native_config.dtype = _DTYPE_MAP[native_config.dtype]
    native_config.device = torch.device(native_config.device)

    # Replace attn func
    if native_config.attn_func not in ATTN_FUNC_MAP:
        raise ValueError(
            f"Unknown attention function name: {native_config.attn_func}. "
            f"Supported attetion functions: {list(ATTN_FUNC_MAP.keys())}"
        )

    native_config.mem_layout = ATTN_FUNC_LAYOUT_MAP[
        native_config.attn_func
    ]  # Set mem layout
    native_config.attn_func_name = native_config.attn_func
    native_config.attn_func = ATTN_FUNC_MAP[native_config.attn_func]
