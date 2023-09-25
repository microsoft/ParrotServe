import inspect
from typing import Optional

from .function import SemanticFunction, logger, ParamType, Parameter
from .placeholder import Placeholder
from .shared_context import SharedContext
from ..orchestration.context import Context


# Annotations of arguments when defining a parrot function.


class Input:
    """Annotate the input."""


class Output:
    """Annotate the output."""


def function(
    register_to_global: bool = True,
    caching_prefix: bool = True,
    emit_py_indent: bool = True,
):
    """A decorator for users to define parrot functions."""

    def create_func(f):
        func_name = f.__name__
        doc_str = f.__doc__

        # Remove the indent of the doc string
        if emit_py_indent:
            possible_indents = ["\t", "    "]
            for indent in possible_indents:
                doc_str = doc_str.replace(indent, "")

        # Parse the function signature (parameters)
        func_sig = inspect.signature(f)
        func_params = []
        for param in func_sig.parameters.values():
            # assert param.annotation in (
            #     Input,
            #     Output,
            # ), "The arguments must be annotated by Input/Output"
            if param.annotation == Input:
                param_typ = ParamType.INPUT
            elif param.annotation == Output:
                param_typ = ParamType.OUTPUT
            else:
                param_typ = ParamType.PYOBJ
            func_params.append(Parameter(name=param.name, typ=param_typ))

        semantic_func = SemanticFunction(
            name=func_name,
            params=func_params,
            cached_prefix=caching_prefix,
            func_body_str=doc_str,
        )

        # controller=None: testing mode
        if register_to_global:
            if SemanticFunction._controller is not None:
                SemanticFunction._controller.register_function(
                    semantic_func, caching_prefix
                )
            else:
                logger.warning("Controller is not set. Not register the function.")

        return semantic_func

    return create_func


def placeholder(
    name: Optional[str] = None,
    content: Optional[str] = None,
) -> Placeholder:
    """Interface to create placeholder."""

    return Placeholder(name, content)


def shared_context(
    engine_name: str,
    parent_context: Optional[Context] = None,
) -> SharedContext:
    """Interface to create a shared context."""

    return SharedContext(engine_name, parent_context)
