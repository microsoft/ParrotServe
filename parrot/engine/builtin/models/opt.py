# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


# NOTE(chaofan): The OPT model is not fine-tuned in chat scenarios, so the
# generated result is not good.
# We recommend use a chat model in real semantic program running scenarios.

# coding=utf-8
#
# Adapted from Huggingface transformers library:
# https://github.com/huggingface/transformers/blob/v4.33-release/src/transformers/models/opt/modeling_opt.py
# Other References:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py
#
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
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

"""PyTorch inference-only OPT model. Input is flattened."""

from typing import Type
import torch
from torch import nn
from transformers import OPTConfig

from .model_utils import hidden_states_postprocess
from .weight_utils import hf_weights_loader
from ..iter_state import IterationState
from .sampler import Sampler
from ..attn_func import AttnFunc
from ...config import BuiltinConfig


ACT_FUNC = {
    "gelu": nn.GELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
}


class OPTLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        layer_idx: int,
        attn_func_cls: Type[AttnFunc],
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # TODO(chaofan): add support for other attention functions
        self.attn_func = attn_func_cls(
            layer_idx=layer_idx,
            scaling=self.scaling,
            head_dim=self.head_dim,
            num_heads=num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        iteration_state: IterationState,
    ) -> torch.Tensor:
        # Shape of hidden_states: [num_tokens, hidden_dims]

        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.chunk(qkv_states, 3, dim=-1)
        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_heads, self.head_dim)
        attn_output = self.attn_func(
            query_states, key_states, value_states, iteration_state
        )
        attn_output = self.out_proj(attn_output)

        return attn_output


class OPTDecoderLayer(nn.Module):
    def __init__(
        self, opt_config: OPTConfig, builtin_config: BuiltinConfig, layer_idx: int
    ):
        super().__init__()
        self.embed_dim = opt_config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=opt_config.num_attention_heads,
            layer_idx=layer_idx,
            attn_func_cls=builtin_config.attn_func,
            bias=opt_config.enable_bias,
        )
        self.do_layer_norm_before = opt_config.do_layer_norm_before
        self.activation_fn = ACT_FUNC[opt_config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=opt_config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(
            self.embed_dim, opt_config.ffn_dim, bias=opt_config.enable_bias
        )
        self.fc2 = nn.Linear(
            opt_config.ffn_dim, self.embed_dim, bias=opt_config.enable_bias
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=opt_config.layer_norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        iteration_state: IterationState,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            iteration_state=iteration_state,
        )

        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):
    def __init__(self, opt_config: OPTConfig, builtin_config: BuiltinConfig):
        super().__init__()
        self.padding_idx = opt_config.pad_token_id
        self.max_target_positions = opt_config.max_position_embeddings
        self.vocab_size = opt_config.vocab_size

        self.embed_tokens = nn.Embedding(
            opt_config.vocab_size, opt_config.word_embed_proj_dim, self.padding_idx
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            opt_config.max_position_embeddings, opt_config.hidden_size
        )

        # Project out & in will be replicated if they exist.
        if opt_config.word_embed_proj_dim != opt_config.hidden_size:
            self.project_out = nn.Linear(
                opt_config.hidden_size, opt_config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if opt_config.word_embed_proj_dim != opt_config.hidden_size:
            self.project_in = nn.Linear(
                opt_config.word_embed_proj_dim, opt_config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if opt_config.do_layer_norm_before and not opt_config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                opt_config.hidden_size,
                elementwise_affine=opt_config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                OPTDecoderLayer(opt_config, builtin_config, i)
                for i in range(opt_config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        iteration_state: IterationState,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        for _, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, iteration_state)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):
    def __init__(self, opt_config: OPTConfig, builtin_config: BuiltinConfig):
        super().__init__()
        self.decoder = OPTDecoder(opt_config, builtin_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        iteration_state: IterationState,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, iteration_state)


class OPTForCausalLM(nn.Module):
    def __init__(self, opt_config: OPTConfig, builtin_config: BuiltinConfig):
        super().__init__()
        self.model = OPTModel(opt_config, builtin_config)
        # Tie lm_head's weight
        self.sampler = Sampler(opt_config, self.model.decoder.embed_tokens.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        iteration_state: IterationState,
    ):
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

        for weight_name, weight_value in hf_weights_loader(model_name_or_path):
            if "lm_head.weight" in weight_name:
                continue
            if weight_name.startswith("decoder."):
                weight_name = "model." + weight_name

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
