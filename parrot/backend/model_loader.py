import torch
from transformers import PretrainedConfig

from .config import RunnerConfig
from ..utils import get_logger
from .models import *


logger = get_logger("Model Loader")


_MODEL_ARCH_MAP = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
}


def load_model(hf_config: PretrainedConfig, runner_config: RunnerConfig):
    model_arch_cls = None
    for arch_name in hf_config.architectures:
        if arch_name in _MODEL_ARCH_MAP:
            model_arch_cls = _MODEL_ARCH_MAP[arch_name]
            runner_config.model_arch = arch_name
            break

    if model_arch_cls is None:
        raise ValueError(
            f"Model architectures {hf_config.architectures} not supported."
            f"Supported models: {_MODEL_ARCH_MAP.keys()}"
        )

    logger.info(
        f"Start loading model {runner_config.model_name} ... (dtype: {runner_config.dtype})"
    )

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(runner_config.dtype)
    model = model_arch_cls(hf_config, runner_config)
    model.load_weights(runner_config.model_name)

    # Move model to device
    model = model.to(runner_config.device)

    torch.set_default_dtype(original_dtype)

    logger.info(f"Model {runner_config.model_name} loaded.")

    return model
