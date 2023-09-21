from typing import List, Optional
from transformers import PretrainedConfig
import torch

from .config import RunnerConfig
from ..utils import get_logger, RecyclePool

logger = get_logger("Mem")


_ARCH_WITH_ROPE = [
    "LlamaForCausalLM",
]


class KVContext:
    """Low-level implementation of Context."""

    def __init__(
        self,
        context_id: int,
        parent_context: Optional["KVContext"],
        kv_cache_manager: RecyclePool,
    ):
        self.context_id = context_id
        self.sub_contexts: List["KVContext"] = []

        # Link with parent context
        self.parent_context = parent_context
        parent_context.sub_contexts.append(self) if parent_context else None
        self.tokens_kv_block_id: List[int] = []
        self.token_ids: List[int] = []

        # KV cache manager i.e. a pool allocator.
        self.kv_cache_manager = kv_cache_manager

        # Flag to indicate whether the context is extended by a Fill primitive.
        # If the context is extended by a Fill primitive recently, the last token
        # will be added the next Generation primitive.
        self.last_extended_by_fill = False

    def __del__(self):
        self.parent_context.sub_contexts.remove(self) if self.parent_context else None
        assert len(self.sub_contexts) == 0, "Sub-contexts should be deleted first."
        for block_id in self.tokens_kv_block_id:
            self.kv_cache_manager.free(block_id)

    def allocate(self, length: int):
        for _ in range(length):
            self.tokens_kv_block_id.append(self.kv_cache_manager.allocate())

    def get_context_len(self) -> int:
        """Return the length of the context."""

        parent_len = self.parent_context.get_context_len() if self.parent_context else 0
        return parent_len + len(self.tokens_kv_block_id)

    def get_context_blocks(self) -> List[int]:
        """Return the context blocks."""

        parent_blocks = (
            self.parent_context.get_context_blocks() if self.parent_context else []
        )
        return parent_blocks + self.tokens_kv_block_id

    def get_last_token_id(self) -> int:
        """Return the last token id."""

        return self.token_ids[-1]


class ModelCacheStorage:
    """Storage for one large language model.

    Including:
    - Key-value cache
    - Cos-sin cache, in rotary embedding models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        runner_config: RunnerConfig,
    ) -> None:
        num_layers = hf_config.num_hidden_layers
        num_blocks = runner_config.num_kv_cache_blocks
        num_heads = hf_config.num_attention_heads
        head_size = hf_config.hidden_size // num_heads
        dtype = runner_config.dtype
        device = runner_config.device

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

        kv_total_size = (
            num_layers
            * num_blocks
            * num_heads
            * head_size
            * self.k_cache.element_size()
            * 2
            / 1024
            / 1024
            / 1024
        )
        logger.info(
            f"Allocated {num_blocks} KV blocks. Total size: {kv_total_size :.2f} GiB"
        )

        # cos / sin cache for rotary embedding models.
        if runner_config.model_arch in _ARCH_WITH_ROPE:
            logger.info(
                f"Model arch {runner_config.model_arch} needs rotary embedding models. "
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
            self.cos_cache = (
                freqs.cos().view(max_seq_len, 1, rotary_size // 2).to(dtype)
            )
            self.sin_cache = (
                freqs.sin().view(max_seq_len, 1, rotary_size // 2).to(dtype)
            )

            cos_sin_total_size = (
                max_seq_len
                * rotary_size
                * 2
                * self.cos_cache.element_size()
                / 1024
                / 1024
            )
            logger.info(
                f"Allocated cos/sin cache. Total size: {cos_sin_total_size :.2f} MiB"
            )
        else:
            logger.info(
                f"Model arch {runner_config.model_arch} doesn't needs rotary embedding models. "
                f"Skip allocating cos/sin cache."
            )

            self.cos_cache = None
            self.sin_cache = None


# Initialize it when the model is loaded.
Model_Cache: Optional[ModelCacheStorage] = None


def init_model_cache_storage(
    hf_config: PretrainedConfig,
    runner_config: RunnerConfig,
) -> None:
    global Model_Cache
    Model_Cache = ModelCacheStorage(hf_config, runner_config)


def get_k_cache(layer_idx: int) -> torch.Tensor:
    global Model_Cache
    assert Model_Cache is not None
    return Model_Cache.k_cache[layer_idx]


def get_v_cache(layer_idx: int) -> torch.Tensor:
    global Model_Cache
    assert Model_Cache is not None
    return Model_Cache.v_cache[layer_idx]


def get_cos_cache() -> torch.Tensor:
    global Model_Cache
    assert Model_Cache is not None
    return Model_Cache.cos_cache


def get_sin_cache() -> torch.Tensor:
    global Model_Cache
    assert Model_Cache is not None
    return Model_Cache.sin_cache
