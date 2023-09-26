import inspect
from typing import Optional

from parrot.orchestration.context import Context

from .function import SemanticFunction, logger, ParamType, Parameter
from .shared_context import SharedContext
from .transforms.prompt_formatter import StandardFormatter, Sequential


# Annotations of arguments when defining a parrot function.


class Input:
    """Annotate the input."""


class Output:
    """Annotate the output."""


def function(
    caching_prefix: bool = True,
    formatter: Sequential = StandardFormatter,
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

        semantic_func = formatter.transform(semantic_func)

        # controller=None: testing mode
        if SemanticFunction._controller is not None:
            SemanticFunction._controller.register_function(
                semantic_func, caching_prefix
            )
        else:
            logger.warning("Controller is not set. Not register the function.")

        return semantic_func

    return create_func


def shared_context(
    engine_name: str,
    parent_context: Optional[Context] = None,
) -> SharedContext:
    """Interface to create a shared context."""

    return SharedContext(engine_name, parent_context)
