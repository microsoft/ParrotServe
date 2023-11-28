# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from .llama import LlamaForCausalLM
from .opt import OPTForCausalLM


MODEL_ARCH_MAP = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
}
