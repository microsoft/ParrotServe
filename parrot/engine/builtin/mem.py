# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional
from transformers import PretrainedConfig
import torch

from parrot.utils import get_logger

from ..config import BuiltinConfig
from .mem_layout import MemLayout

logger = get_logger("Mem")


_ARCH_WITH_ROPE = [
    "LlamaForCausalLM",
]


class ModelCacheStorage:
    """Storage for one large language model.

    Including:
    - Key-value cache
    - Cos-sin cache, in rotary embedding models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        builtin_config: BuiltinConfig,
    ) -> None:
        num_layers = hf_config.num_hidden_layers
        num_blocks = builtin_config.num_kv_cache_blocks
        block_size = builtin_config.block_size
        num_heads = hf_config.num_attention_heads
        head_size = hf_config.hidden_size // num_heads
        dtype = builtin_config.dtype
        device = builtin_config.device

        if builtin_config.mem_layout == MemLayout.NORMAL:
            assert block_size == 1, "Block size must be 1 for normal layout."

            self.k_cache = torch.empty(
                [num_layers, num_blocks, num_heads, head_size],
                dtype=dtype,
                device=device,
            )

            self.v_cache = torch.empty(
                [num_layers, num_blocks, num_heads, head_size],
                dtype=dtype,
                device=device,
            )
        elif builtin_config.mem_layout == MemLayout.BLOCK:
            self.k_cache = torch.empty(
                [num_layers, num_blocks, num_heads, head_size, block_size],
                dtype=dtype,
                device=device,
            )

            self.v_cache = torch.empty(
                [num_layers, num_blocks, num_heads, head_size, block_size],
                dtype=dtype,
                device=device,
            )
        elif builtin_config.mem_layout == MemLayout.VLLM:
            self.v_cache = torch.empty(
                [num_layers, num_blocks, num_heads, head_size, block_size],
                dtype=dtype,
                device=device,
            )

            x = 16 // self.v_cache.element_size()

            self.k_cache = torch.empty(
                [num_layers, num_blocks, num_heads, head_size // x, block_size, x],
                dtype=dtype,
                device=device,
            )

        kv_total_size = (
            num_layers
            * num_blocks
            * block_size
            * num_heads
            * head_size
            * self.k_cache.element_size()
            * 2
            / 1024
            / 1024
            / 1024
        )

        logger.info(
            f"Allocated {num_blocks} KV blocks. "
            f"Mem Layout: {builtin_config.mem_layout.name}. "
            f"Per block size: {block_size}. "
            f"Total size: {kv_total_size :.2f} GiB."
        )

        # cos / sin cache for rotary embedding models.
        if builtin_config.model_arch in _ARCH_WITH_ROPE:
            logger.info(
                f"Model arch {builtin_config.model_arch} needs rotary embedding models. "
                f"Allcoating cos/sin cache ..."
            )

            max_seq_len = hf_config.max_position_embeddings
            # self.cos_cache = torch.empty(
            #     [max_seq_len, 1, head_size // 2],
            #     dtype=dtype,
            #     device=device,
            # )

            # self.sin_cache = torch.empty(
            #     [max_seq_len, 1, head_size // 2],
            #     dtype=dtype,
            #     device=device,
            # )

            # Requires transformers > 4.32.0
            rope_theta = rope_theta = getattr(hf_config, "rope_theta", 10000)
            rotary_size = head_size
            inv_freq = 1.0 / (
                rope_theta
                ** (
                    torch.arange(0, rotary_size, 2, device=device).float() / rotary_size
                )
            )
            t = torch.arange(max_seq_len, dtype=inv_freq.dtype, device=device)
            freqs = torch.outer(t, inv_freq)
            # self.cos_cache = (
            #     freqs.cos().view(max_seq_len, 1, rotary_size // 2).to(dtype)
            # )
            # self.sin_cache = (
            #     freqs.sin().view(max_seq_len, 1, rotary_size // 2).to(dtype)
            # )

            self.cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
            self.cos_sin_cache = self.cos_sin_cache.to(dtype)

            cos_sin_total_size = (
                max_seq_len
                * rotary_size
                * 2
                * self.cos_sin_cache.element_size()
                / 1024
                / 1024
            )
            logger.info(
                f"Allocated cos/sin cache for max_seq_len {max_seq_len}. "
                f"Total size: {cos_sin_total_size :.2f} MiB"
            )
        else:
            logger.info(
                f"Model arch {builtin_config.model_arch} doesn't needs rotary embedding models. "
                f"Skip allocating cos/sin cache."
            )

            # self.cos_cache = None
            # self.sin_cache = None


# Initialize it when the model is loaded.
Model_Cache: Optional[ModelCacheStorage] = None


def init_model_cache_storage(
    hf_config: PretrainedConfig,
    builtin_config: BuiltinConfig,
) -> None:
    global Model_Cache
    Model_Cache = ModelCacheStorage(hf_config, builtin_config)


def get_k_cache(layer_idx: int) -> torch.Tensor:
    global Model_Cache
    assert Model_Cache is not None
    return Model_Cache.k_cache[layer_idx]


def get_v_cache(layer_idx: int) -> torch.Tensor:
    global Model_Cache
    assert Model_Cache is not None
    return Model_Cache.v_cache[layer_idx]


# def get_cos_cache() -> torch.Tensor:
#     global Model_Cache
#     assert Model_Cache is not None
#     return Model_Cache.cos_cache


# def get_sin_cache() -> torch.Tensor:
#     global Model_Cache
#     assert Model_Cache is not None
#     return Model_Cache.sin_cache


def get_cos_sin_cache() -> torch.Tensor:
    global Model_Cache
    assert Model_Cache is not None
    return Model_Cache.cos_sin_cache
