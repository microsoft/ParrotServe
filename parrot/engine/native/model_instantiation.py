import torch
from transformers import PretrainedConfig
import contextlib

from parrot.utils import get_logger

from ..config import NativeConfig
from .models import MODEL_ARCH_MAP


logger = get_logger("Model Instantiation")


@contextlib.contextmanager
def model_instantiation_context(
    model_name: str, native_config: NativeConfig, dummy_weight_init: bool
):
    """Provide a context for instantiating models.

    Including:
    - Set dtype
    - Disable weight initialization for faster loading (mainly in Linear)
    """

    logger.info(
        f"Start instantiating model {model_name} ... (dtype: {native_config.dtype})"
    )

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(native_config.dtype)
    if not dummy_weight_init:
        original_reset_parameters = torch.nn.Linear.reset_parameters
        torch.nn.Linear.reset_parameters = (
            lambda self: None
        )  # This is a very hacky way to disable weight initialization

    yield

    torch.set_default_dtype(original_dtype)
    if not dummy_weight_init:
        torch.nn.Linear.reset_parameters = original_reset_parameters

    logger.info(f"Model {model_name} instantiated. Weights loaded.")


@torch.no_grad()
def instantiate_model(
    model_name: str, hf_config: PretrainedConfig, native_config: NativeConfig
):
    # Get the model architecture
    model_arch_cls = None
    for arch_name in hf_config.architectures:
        if arch_name in MODEL_ARCH_MAP:
            model_arch_cls = MODEL_ARCH_MAP[arch_name]
            native_config.model_arch = arch_name
            break

    if model_arch_cls is None:
        raise ValueError(
            f"Model architectures {hf_config.architectures} not supported."
            f"Supported models: {MODEL_ARCH_MAP.keys()}"
        )

    with model_instantiation_context(model_name, native_config, False):
        model = model_arch_cls(hf_config, native_config)
        model.load_weights(model_name)

        # Move model to device
        model = model.to(native_config.device)

    return model
