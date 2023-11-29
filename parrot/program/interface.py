# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import inspect
import collections
from typing import Optional, List

from parrot.protocol.sampling_config import SamplingConfig
from parrot.protocol.annotation import DispatchAnnotation
from parrot.utils import get_logger, change_signature

from .semantic_variable import SemanticVariable
from .function import SemanticFunction, NativeFunction, ParamType, Parameter
from .transforms.prompt_formatter import standard_formatter, Sequential, FuncMutator


logger = get_logger("Interface")


# Annotations of arguments when defining a parrot function.


class Input:
    """Annotate the Input semantic variable in the Parrot function signature."""


class Output:
    """Annotate the Output semantic varialbe in the Parrot function signature."""

    def __init__(
        self,
        sampling_config: SamplingConfig = SamplingConfig(),
        dispatch_annotation: DispatchAnnotation = DispatchAnnotation(),
    ):
        self.sampling_config = sampling_config
        self.dispatch_annotation = dispatch_annotation


def semantic_function(
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

        # print(doc_str)

        # Parse the function signature (parameters)
        func_sig = inspect.signature(f)
        func_params = []
        for param in func_sig.parameters.values():
            # assert param.annotation in (
            #     Input,
            #     Output,
            # ), "The arguments must be annotated by Input/Output"

            kwargs = {}

            if param.annotation == Input:
                param_typ = ParamType.INPUT_LOC
            elif param.annotation == Output:
                # Default output loc
                param_typ = ParamType.OUTPUT_LOC
                kwargs = {
                    "sampling_config": SamplingConfig(),
                    "dispatch_annotation": DispatchAnnotation(),
                }
            elif param.annotation.__class__ == Output:
                # Output loc with sampling config
                param_typ = ParamType.OUTPUT_LOC
                kwargs = {
                    "sampling_config": param.annotation.sampling_config,
                    "dispatch_annotation": param.annotation.dispatch_annotation,
                }
            else:
                param_typ = ParamType.INPUT_PYOBJ
            func_params.append(Parameter(name=param.name, typ=param_typ, **kwargs))

        semantic_func = SemanticFunction(
            name=func_name,
            params=func_params,
            func_body_str=doc_str,
            # Func Metadata
            models=models,
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


def native_function(
    timeout: float = 0.1,
):
    """A decorator for users to define parrot functions."""

    def create_func(f):
        func_name = f.__name__

        # Parse the function signature (parameters)
        func_sig = inspect.signature(f)
        return_annotations = func_sig.return_annotation
        func_params = []

        # Update annotations for the pyfunc
        new_params_anotations = []
        new_return_annotations = []

        for param in func_sig.parameters.values():
            if param.annotation == Input:
                param_typ = ParamType.INPUT_LOC
                new_params_anotations.append(
                    inspect.Parameter(
                        param.name,
                        param.kind,
                        default=param.default,
                        annotation=str,
                    )
                )
            elif param.annotation == Output:
                raise ValueError(
                    "Please put Output annotation in the return type in native function."
                )
            elif param.annotation.__class__ == Output:
                raise ValueError(
                    "Please put Output annotation in the return type in native function."
                )
            else:
                param_typ = ParamType.INPUT_PYOBJ
                new_params_anotations.append(
                    inspect.Parameter(
                        param.name,
                        param.kind,
                        default=param.default,
                        annotatioin=param.annotation,
                    )
                )
            func_params.append(Parameter(name=param.name, typ=param_typ))

        if return_annotations == inspect.Signature.empty:
            raise ValueError("Native function must return at least one P.Output.")
        elif not isinstance(return_annotations, collections.abc.Iterable):
            return_annotations = [
                return_annotations,
            ]
            # raise ValueError("Native function can only return one P.Output.")

        ret_counter = 0
        for annotation in return_annotations:
            if annotation == Output:
                func_params.append(
                    Parameter(name=f"ret_{ret_counter}", typ=ParamType.OUTPUT_LOC)
                )
                ret_counter += 1
                new_return_annotations.append(str)
            elif annotation.__class__ == Output:
                # Output loc with sampling config
                raise ValueError(
                    "Native function does not support annotate Output variables."
                )
            else:
                raise ValueError("Native function can only return P.Output")

        change_signature(f, new_params_anotations, new_return_annotations)

        native_func = NativeFunction(
            name=func_name,
            pyfunc=f,
            params=func_params,
            # Func Metadata
            timeout=timeout,
        )

        return native_func

    return create_func


def variable(
    name: Optional[str] = None, content: Optional[str] = None
) -> SemanticVariable:
    """Let user construct Semantic Variable explicitly."""

    return SemanticVariable(name, content)


# def shared_context(
#     engine_name: str,
#     parent_context: Optional[Context] = None,
# ) -> SharedContext:
#     """Interface to create a shared context."""

#     return SharedContext(engine_name, parent_context)
