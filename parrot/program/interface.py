# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import inspect
from typing import Optional, List

from parrot.protocol.sampling_config import SamplingConfig
from parrot.utils import get_logger

from .function import SemanticFunction, ParamType, Parameter
from .transforms.prompt_formatter import standard_formatter, Sequential, FuncMutator


logger = get_logger("Interface")


# Annotations of arguments when defining a parrot function.


class Input:
    """Annotate the input."""


class Output:
    """Annotate the output."""

    def __init__(self, *args, **kwargs):
        self.sampling_config = SamplingConfig(*args, **kwargs)


def function(
    models: List[str] = [],
    cache_prefix: bool = True,
    remove_pure_fill: bool = True,
    formatter: Optional[Sequential] = standard_formatter,
    conversation_template: Optional[FuncMutator] = None,
):
    """A decorator for users to define parrot functions."""

    def create_func(f):
        func_name = f.__name__
        doc_str = f.__doc__

        # Parse the function signature (parameters)
        func_sig = inspect.signature(f)
        func_params = []
        for param in func_sig.parameters.values():
            # assert param.annotation in (
            #     Input,
            #     Output,
            # ), "The arguments must be annotated by Input/Output"
            sampling_config: Optional[SamplingConfig] = None
            if param.annotation == Input:
                param_typ = ParamType.INPUT_LOC
            elif param.annotation == Output:
                # Default output loc
                param_typ = ParamType.OUTPUT_LOC
                sampling_config = SamplingConfig()
            elif param.annotation.__class__ == Output:
                # Output loc with sampling config
                param_typ = ParamType.OUTPUT_LOC
                sampling_config = param.annotation.sampling_config
            else:
                param_typ = ParamType.INPUT_PYOBJ
            func_params.append(
                Parameter(
                    name=param.name, typ=param_typ, sampling_config=sampling_config
                )
            )

        semantic_func = SemanticFunction(
            name=func_name,
            params=func_params,
            models=models,
            func_body_str=doc_str,
            # Func Metadata
            cache_prefix=cache_prefix,
            remove_pure_fill=remove_pure_fill,
        )

        if formatter is not None:
            semantic_func = formatter.transform(semantic_func)
        if conversation_template is not None:
            logger.warning(
                f"Use a conversation template {conversation_template.__class__.__name__} to "
                "transform the function. This only works well for requests which are dispatched "
                "to engines with the corresponding models."
            )
            semantic_func = conversation_template.transform(semantic_func)

        return semantic_func

    return create_func


# def shared_context(
#     engine_name: str,
#     parent_context: Optional[Context] = None,
# ) -> SharedContext:
#     """Interface to create a shared context."""

#     return SharedContext(engine_name, parent_context)
