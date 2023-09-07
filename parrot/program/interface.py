import inspect
from typing import Optional

from .function import ParrotFunction
from .placeholder import Placeholder

from ..global_init import parrot_global_ctrl, global_tokenized_storage

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
        if emit_py_indent:
            possible_indents = ["\t", "    "]
            for indent in possible_indents:
                doc_str = doc_str.replace("\n" + indent, " ")
                doc_str = doc_str.replace(indent, "")
        func_sig = inspect.signature(f)
        func_args = []
        for param in func_sig.parameters.values():
            assert param.annotation in (
                Input,
                Output,
            ), "The arguments must be annotated by Input/Output"
            func_args.append((param.name, param.annotation == Output))
        parrot_func = ParrotFunction(
            name=func_name,
            func_body_str=doc_str,
            func_args=func_args,
        )

        if register_to_global:
            parrot_global_ctrl.register_function(
                parrot_func, global_tokenized_storage if caching_prefix else None
            )

        return parrot_func

    return create_func


def placeholder(name: Optional[str] = None, content: Optional[str] = None):
    """Interface to create placeholder."""

    return Placeholder(name, content)
