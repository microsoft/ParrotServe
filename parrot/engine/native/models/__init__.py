from .llama import LlamaForCausalLM
from .opt import OPTForCausalLM


MODEL_ARCH_MAP = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
}
