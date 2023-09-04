import inspect
from typing import Optional

from .function import ParrotFunction
from .placeholder import Placeholder

# Annotations of arguments when defining a parrot function.


class Input:
    """Annotate the input."""


class Output:
    """Annotate the output."""


def function(
    caching_prefix: bool = True,
    emit_py_indent: bool = True,
):
    """A decorator for users to define parrot functions."""

    def parse(f):
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
        return ParrotFunction(
            name=func_name,
            func_body_str=doc_str,
            func_args=func_args,
        )

    return parse


def placeholder(name: str, content: Optional[str] = None):
    """Interface to create placeholder."""

    return Placeholder(name, content)
