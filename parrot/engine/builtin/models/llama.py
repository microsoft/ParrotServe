# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

# coding=utf-8
#
# Adapted from Huggingface transformers library:
# https://github.com/huggingface/transformers/blob/v4.33-release/src/transformers/models/llama/modeling_llama.py
# Other References:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
#
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch inference-only LLaMA model. Input is flattened."""

from typing import Type
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaMLP,
)

from .model_utils import hidden_states_postprocess
from .weight_utils import hf_weights_loader
from ..iter_state import IterationState
from ..mem import get_cos_sin_cache
from ...config import BuiltinConfig
from .sampler import Sampler
from ..attn_func import AttnFunc

# from ..kernels import rotary_embedding, rmsnorm_forward

from ..kernels import vllm_rms_norm, vllm_rotary_emb


class LlamaRMSNorm(nn.Module):
    """Wrapper for RMSNorm."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return vllm_rms_norm(x, self.weight, self.eps)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        attn_func_cls: Type[AttnFunc],
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # Currently don't support MQA/GQA
        # self.num_key_value_heads = config.num_key_value_heads
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # TODO(chaofan): add support for other attention functions
        self.attn_func = attn_func_cls(
            layer_idx=layer_idx,
            scaling=self.scaling,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        iteration_state: IterationState,
    ) -> torch.Tensor:
        # Shape of hidden_states: [num_tokens, hidden_dims]

        # get query, key, value
        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.chunk(qkv_states, 3, dim=-1)

        # Should we fuse?
        # rotary_embedding(
        #     query_states, iteration_state.cos_buffer, iteration_state.sin_buffer
        # )
        # rotary_embedding(
        #     key_states, iteration_state.cos_buffer, iteration_state.sin_buffer
        # )

        cos_sin_cache = get_cos_sin_cache()
        vllm_rotary_emb(
            positions,
            query_states,
            key_states,
            cos_sin_cache,
            head_size=self.head_dim,
        )

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_heads, self.head_dim)

        attn_output = self.attn_func(
            query_states, key_states, value_states, iteration_state
        )
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self, config: LlamaConfig, builtin_config: BuiltinConfig, layer_idx: int
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(
            config=config,
            layer_idx=layer_idx,
            attn_func_cls=builtin_config.attn_func,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        iteration_state: IterationState,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, iteration_state)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig, builtin_config: BuiltinConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, builtin_config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        iteration_state: IterationState,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for _, layer in enumerate(self.layers):
            # print(f"Layer: {_}. Start forward.", flush=True)
            hidden_states = layer(hidden_states, positions, iteration_state)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig, builtin_config: BuiltinConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, builtin_config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sampler = Sampler(config, self.lm_head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        iteration_state: IterationState,
    ):
        # print(positions)
        hidden_states = self.model(input_ids, positions, iteration_state)

        fill_hidden_states, gen_hidden_states = hidden_states_postprocess(
            hidden_states, iteration_state
        )
        next_tokens = self.sampler(
            gen_hidden_states, iteration_state.generation_sampling_config
        )
        return fill_hidden_states, next_tokens

    def load_weights(self, model_name_or_path: str):
        state_dict = self.state_dict()
        # print(state_dict.keys())

        for weight_name, weight_value in hf_weights_loader(model_name_or_path):
            if "rotary_emb.inv_freq" in weight_name:
                continue

            # Handle qkv_proj
            is_qkv_weight = False
            for stride_id, qkv_proj_name in enumerate(["q_proj", "k_proj", "v_proj"]):
                if qkv_proj_name not in weight_name:
                    continue
                param = state_dict[weight_name.replace(qkv_proj_name, "qkv_proj")]
                shard_size = param.shape[0] // 3

                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == weight_value.shape
                param_slice.copy_(weight_value)
                is_qkv_weight = True
                break

            if not is_qkv_weight:
                param = state_dict[weight_name]
                param.copy_(weight_value)
            # print(f"{name} loaded.")
