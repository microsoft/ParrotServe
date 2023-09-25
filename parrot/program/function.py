from enum import Enum
from typing import List, Dict, Type, Optional, Any
import regex as re
from dataclasses import dataclass

from parrot.utils import get_logger

from .placeholder import Placeholder


logger = get_logger("Function")


@dataclass
class FunctionPiece:
    """A piece in the function body."""

    idx: int


@dataclass
class Constant(FunctionPiece):
    """Constant text."""

    text: str


@dataclass
class Prefix(Constant):
    """Prefix in a function."""


class ParamType(Enum):
    """Type of a parameter."""

    INPUT = 0
    OUTPUT = 1
    PYOBJ = 2


@dataclass
class Parameter:
    name: str
    typ: ParamType

    @property
    def is_output(self) -> bool:
        return self.typ == ParamType.OUTPUT


@dataclass
class ParameterLoc(FunctionPiece):
    """An input/output location in the function."""

    param: Parameter


def parse_func_body(
    body_str: str, params_map: Dict[str, Parameter]
) -> List[FunctionPiece]:
    PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]*}}"
    pattern = re.compile(PLACEHOLDER_REGEX)
    iterator = pattern.finditer(body_str)
    last_pos = 0

    ret: List[FunctionPiece] = []

    def push_to_body(piece_cls: Type[FunctionPiece], **kwargs):
        idx = len(ret)
        ret.append(piece_cls(idx=idx, **kwargs))

    last_output_loc_idx = -1

    for match in iterator:
        # Constant
        chunk = body_str[last_pos : match.start()]
        if chunk != "":
            if last_pos == 0:
                push_to_body(Prefix, text=chunk)
            else:
                push_to_body(Constant, text=chunk)

        param_name = body_str[match.start() + 2 : match.end() - 2]
        assert param_name in params_map, f"Parse failed: {param_name} is not defined."
        param = params_map[param_name]
        assert not (
            param.is_output and isinstance(ret[-1], ParameterLoc)
        ), "Output loc can't be adjacent to another loc."
        push_to_body(ParameterLoc, param=param)

        if param.is_output:
            last_output_loc_idx = len(ret) - 1

        last_pos = match.end()

    # if last_pos < len(body_str):
    #     push_to_body(Constant, body_str[last_pos:])

    # NOTE(chaofan): we prune all pieces after the last output loc.
    # The following code is also correct for last_output_loc_idx == -1.
    ret = ret[: last_output_loc_idx + 1]

    return ret


class SemanticFunction:
    """Parrot's semantic function is a simplified abstraction of the "general" semantic function,
    which is used as examples when we play in the Parrot project.

    An example:
        ```
        Tell me a joke about {{topic}}. The joke must contains the
        following keywords: {{keyword}}. The following is the joke: {{joke}}.
        And giving a short explanation to show that why it is funny. The following is the
        explanation for the joke above: {{explanation}}.
        ```
    """

    _controller: Optional["Controller"] = None
    _executor: Optional["Executor"] = None

    def __init__(
        self,
        name: str,
        func_body_str: str,
        params: List[Parameter],
        cached_prefix: bool,
    ):
        """For semantic function, function body is just a prompt template.
        After parsed, it turns to be a list of function pieces.
        """

        self.name = name
        self.params = params
        self.cached_prefix = cached_prefix
        self.params_map = dict([(param.name, param) for param in self.params])
        self.body: List[FunctionPiece] = parse_func_body(func_body_str, self.params_map)

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        """Calling a parrot function will not execute it immediately.
        Instead, this will submit the call to the executor."""

        promise = Promise(self, None, *args, **kwargs)
        if SemanticFunction._executor is not None:
            SemanticFunction._executor.submit(promise)
        else:
            logger.warning(
                "Executor is not set, will not submit the promise. "
                "(Please ensure run a Parrot function in a running context, e.g. using env.parrot_run_aysnc)"
            )


class Promise:
    """Promise is a function call including the functions and bindings (param name ->
    placeholder)."""

    def __init__(
        self,
        func: SemanticFunction,
        shared_context_handler: Optional["SharedContextHandler"] = None,
        *args,
        **kwargs,
    ):
        self.func = func
        self.bindings: Dict[str, Any] = {}
        self.shared_context_handler = shared_context_handler

        for i, arg_value in enumerate(args):
            self._set_value(self.func.params[i], arg_value, self.bindings)

        for name, arg_value in kwargs.items():
            assert (
                name not in self.bindings
            ), f"Function {self.func.name} got multiple values for argument {name}"
            assert (
                name in self.func.params_map
            ), f"Function {self.func.name} got an unexpected keyword argument {name}"
            self._set_value(self.func.params_map[name], arg_value, self.bindings)

    @staticmethod
    def _set_value(param: Parameter, value: Any, bindings: Dict[str, Any]):
        if param.typ != ParamType.PYOBJ:
            assert isinstance(value, Placeholder), (
                f"Argument {param.name} should be a placeholder, "
                f"but got {type(value)}: {value}"
            )
        bindings[param.name] = value
