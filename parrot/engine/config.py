# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Literal, Optional, Dict, Union
from dataclasses import dataclass
import torch

from parrot.constants import (
    FILL_NO_CHUNK,
    DEFAULT_SERVER_HOST,
    DEFAULT_ENGINE_SERVER_PORT,
    ENGINE_TYPE_BUILTIN,
    ENGINE_TYPES,
)

from .builtin.mem_layout import MemLayout, ATTN_FUNC_LAYOUT_MAP

from .openai.api_endpoint import Endpoint, ENDPOINT_MAP


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class BuiltinConfig:
    num_kv_cache_blocks: int
    attn_func: Union[str, "AttnFunc"]
    dtype: Union[Literal["float16", "float32"], torch.dtype] = "float16"
    device: Union[str, torch.device] = "cuda"  # cpu, cuda, cuda:x
    block_size: int = 1
    max_seq_len: Optional[int] = None  # Override the original model length
    attn_func_name: Optional[str] = None
    mem_layout: Optional["MemLayout"] = None
    model_arch: Optional[str] = None

    def __post_init__(self):
        # Replace dtype and device
        self.dtype_str = self.dtype
        self.device_str = self.device
        self.dtype = _DTYPE_MAP[self.dtype]
        self.device = torch.device(self.device)

        # Replace attn func
        self.mem_layout = ATTN_FUNC_LAYOUT_MAP[self.attn_func]  # Set mem layout
        self.attn_func_name = self.attn_func
        self.attn_func = self._get_attn_func(self.attn_func)


@dataclass
class MLCConfig:
    model_path: str
    lib_path: str
    device: str = "cuda"  # 'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto'


@dataclass
class OpenAIConfig:
    api_key: str
    api_endpoint: Union[str, Endpoint]
    base_url: Optional[str] = None

    # Azure OpenAI related
    is_azure: bool = False
    azure_api_version: str = "2023-07-01-preview"
    azure_endpoint: str = "https://example-endpoint.openai.azure.com"

    def __post_init__(self):
        if self.api_endpoint not in ENDPOINT_MAP:
            raise ValueError(
                f"Unknown endpoint name: {self.api_endpoint}. "
                f"Supported endpoints: {list(ENDPOINT_MAP.keys())}"
            )
        self.api_endpoint = ENDPOINT_MAP[self.api_endpoint]


@dataclass
class HuggingFaceConfig:
    dtype: Literal["float16", "float32"] = "float16"
    device: str = "cuda"


@dataclass
class SchedulerConfig:
    max_batch_size: int
    max_num_batched_tokens: int
    max_total_tokens: int
    policy: Literal["fifo", "tgi"] = "fifo"


@dataclass
class EngineConfig:
    # The model used in this engine.
    # - For open source LLMs, the model name must follow the format in HuggingFace,
    #   e.g. facebook/opt-13b;
    # - For OpenAI API, the model name is the one used in OpenAI API,
    #   i.e. deployment name.
    model: str = "unknown"

    # Host and port in engine server.
    host: str = DEFAULT_SERVER_HOST
    port: int = DEFAULT_ENGINE_SERVER_PORT

    # The name of engine.
    engine_name: str = "unknown"

    # The type of engine.
    engine_type: str = ENGINE_TYPE_BUILTIN

    # Random seed for reproduction.
    random_seed: int = 0

    # The tokenizer. Some engines (e.g. OpenAI) do not need tokenizer.
    # For local LLMs, the tokenizer name must follow the format of
    # HugoingFace tokenizer name, e.g. facebook/opt-13b.
    tokenizer: str = "unknown"
    fill_chunk_size: int = FILL_NO_CHUNK

    # The folowing configs are forwarded from sub configs, and is not
    # required to be set in the engine config.

    # Forward from runner config
    dtype: Literal["float16", "float32"] = "float16"
    device: str = "cuda"  # cpu, cuda, cuda:x

    # Max threads the engine can handle.
    threads_capacity: int = 256

    # For non-builtin engines, it's useless.
    tokens_capacity: int = 262144

    @classmethod
    def verify_config(cls, config: Dict) -> bool:
        """Verify the engine config."""

        if "instance" not in config or "scheduler" not in config:
            return False

        # for field in cls.__dataclass_fields__:
        #     if field in runner_keys:
        #         continue

        #     if field not in config:
        #         return False

        # Check Literal
        if config["engine_type"] not in ENGINE_TYPES:
            return False

        return True
