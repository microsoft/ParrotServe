# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py
# Copyright 2023 The vLLM team.
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

""" PyTorch inference-only OPT model. Input is flattened."""

from typing import Optional, List

import torch
from torch import nn
import numpy as np
from transformers import OPTConfig

from xformers import ops as xops

from .weight_utils import hf_model_weights_iterator
from .state_cache import StateCache
from ..entity import InputMetadata
from .sampler import GreedySampler

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
        has_cache: bool,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.cache = StateCache() if has_cache else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: InputMetadata,
    ) -> torch.Tensor:
        # Shape of hidden_states: [num_tokens, hidden_dims]

        # get query proj
        query_states = self.q_proj(hidden_states)

        if self.cache:
            self.cache.put(hidden_states, metadata)
            hidden_states = torch.empty(
                (np.sum(metadata.lens), self.embed_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self.cache.get(metadata, placeholder=hidden_states)
            attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                q_seqlen=metadata.prefill_lens,
                kv_seqlen=metadata.lens,
            )
        else:
            attn_bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(
                metadata.lens
            )

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # reshape
        query_states = query_states.view(1, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(1, -1, self.num_heads, self.head_dim)
        value_states = value_states.view(1, -1, self.num_heads, self.head_dim)

        attn_output = xops.memory_efficient_attention_forward(
            query_states,
            key_states,
            value_states,
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scaling,
            op=xops.fmha.cutlass.FwOp(),
        )

        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig, has_cache: bool):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = OPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            has_cache=has_cache,
        )
        self.has_cache = has_cache
        self.do_layer_norm_before = config.do_layer_norm_before
        self.activation_fn = ACT_FUNC[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata: InputMetadata,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            metadata=metadata,
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
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                OPTDecoderLayer(config, has_cache=(i != 0))
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: InputMetadata,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        pos_embeds = self.embed_positions(positions)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        pruned = False
        for layer in self.layers:
            if not pruned and layer.has_cache:
                idx = 0
                indicies: List[int] = []
                for i in range(len(metadata.seq_ids)):
                    indicies.extend(
                        range(idx + metadata.cached_lens[i], idx + metadata.lens[i])
                    )
                    idx += metadata.lens[i]
                hidden_states = hidden_states[indicies]
                pruned = True
            hidden_states = layer(hidden_states, metadata)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        return hidden_states


class OPTModel(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.decoder = OPTDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: InputMetadata,
    ) -> torch.Tensor:
        return self.decoder(input_ids, positions, metadata)


class OPTForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = OPTModel(config)
        # Tie lm_head's weight
        self.sampler = GreedySampler(config, self.model.decoder.embed_tokens.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: InputMetadata,
    ):
        hidden_states = self.model(input_ids, positions, metadata)
        next_tokens = self.sampler(hidden_states, metadata)
        return next_tokens

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        use_np_cache: bool = False,
    ):
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache
        ):
            if "lm_head.weight" in name:
                continue
            if name.startswith("decoder."):
                name = "model." + name
            param = state_dict[name]
            param.copy_(loaded_weight)
            # print(f"{name} loaded.")
