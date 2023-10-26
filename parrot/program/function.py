from enum import Enum
from typing import List, Dict, Type, Optional, Any, Set, Union, Tuple
import pickle
import regex as re
from dataclasses import dataclass

from parrot.utils import get_logger
from parrot.protocol.sampling_config import SamplingConfig

from .future import Future


logger = get_logger("Function")


@dataclass
class FunctionPiece:
    """A piece in the function body."""

    idx: int


@dataclass
class Constant(FunctionPiece):
    """Constant text."""

    text: str


class ParamType(Enum):
    """Type of a parameter."""

    INPUT_LOC = 0
    OUTPUT_LOC = 1
    INPUT_PYOBJ = 2


@dataclass
class Parameter:
    name: str
    typ: ParamType
    sampling_config: Optional[SamplingConfig] = None

    @property
    def is_output(self) -> bool:
        return self.typ == ParamType.OUTPUT_LOC


@dataclass
class FunctionMetadata:
    """Metadata of a function."""

    cache_prefix: bool
    models: List[str]


@dataclass
class ParameterLoc(FunctionPiece):
    """An input/output location in the function."""

    param: Parameter


def push_to_body(piece_cls: Type[FunctionPiece], body: List[FunctionPiece], **kwargs):
    idx = len(body)
    body.append(piece_cls(idx=idx, **kwargs))


def parse_func_body(
    body_str: str, params_map: Dict[str, Parameter]
) -> List[FunctionPiece]:
    PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]*}}"
    pattern = re.compile(PLACEHOLDER_REGEX)
    iterator = pattern.finditer(body_str)
    last_pos = 0

    ret: List[FunctionPiece] = []

    last_output_loc_idx = -1
    outputs: Set[str] = set()

    for match in iterator:
        # Constant
        chunk = body_str[last_pos : match.start()]
        if chunk != "":
            push_to_body(Constant, ret, text=chunk)

        param_name = body_str[match.start() + 2 : match.end() - 2]
        assert param_name in params_map, f"Parse failed: {param_name} is not defined."
        param = params_map[param_name]
        if param.is_output:
            assert not (
                isinstance(ret[-1], ParameterLoc)
            ), "Output loc can't be adjacent to another loc."
            assert not param.name in outputs, "Output param can't be used twice."
            outputs.add(param.name)
        push_to_body(ParameterLoc, ret, param=param)

        if param.is_output:
            last_output_loc_idx = len(ret) - 1

        last_pos = match.end()

    # if last_pos < len(body_str):
    #     push_to_body(Constant, ret, body_str[last_pos:])

    # NOTE(chaofan): we prune all pieces after the last output loc.
    # The following code is also correct for last_output_loc_idx == -1.
    ret = ret[: last_output_loc_idx + 1]

    return ret


@dataclass
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

    _virtual_machine_env: Optional["VirtualMachine"] = None

    def __init__(
        self,
        name: str,
        params: List[Parameter],
        func_body_str: Optional[str] = None,
        func_body: Optional[List[FunctionPiece]] = None,
        **kwargs,
    ):
        """For semantic function, function body is just a prompt template.
        After parsed, it turns to be a list of function pieces.
        """

        # ---------- Basic Info ----------
        self.name = name
        self.params = params
        self.params_map = dict([(param.name, param) for param in self.params])
        self.inputs = [
            param for param in self.params if param.typ != ParamType.OUTPUT_LOC
        ]
        self.outputs = [
            param for param in self.params if param.typ == ParamType.OUTPUT_LOC
        ]
        if func_body_str is not None:
            self.body: List[FunctionPiece] = parse_func_body(
                func_body_str, self.params_map
            )
        elif func_body is not None:
            self.body = func_body
        else:
            raise ValueError("Either func_body_str or func_body should be provided.")

        self.metadata = FunctionMetadata(**kwargs)

    def __call__(
        self, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> Union[Future, Tuple[Future, ...], "SemanticCall"]:
        """Call to a semantic function.

        Some notes:
        - Calling a parrot function will not execute it immediately.
          Instead, this will submit the call to OS.

        - The return value is a list of Future objects, which can be used to get the
          output contents or passed to other functions.

        - Caller should provide all the input arguments, including INPUT_LOC and INPUT_PYOBJ.

        - The INPUT_PYOBJ arguments should be Python objects, which will be turns to a string
          using __str__ method.
        """

        call = SemanticCall(self, *args, **kwargs)
        if SemanticFunction._virtual_machine_env is not None:
            SemanticFunction._virtual_machine_env._submit_call(call)
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return call

        # Unpack the output futures
        if len(call.output_futures) == 1:
            return call.output_futures[0]
        return tuple(call.output_futures)

    @property
    def prefix(self) -> Constant:
        """Get the prefix of the function body."""
        return self.body[0]

    def display(self) -> str:
        """Display the function body."""
        return "".join(
            [
                piece.text
                if isinstance(piece, Constant)
                else "{{" + piece.param.name + "}}"
                for piece in self.body
            ]
        )


class SemanticCall:
    """A call to a semantic function."""

    def __init__(
        self,
        func: SemanticFunction,
        *args,
        **kwargs,
    ):
        self.func = func
        self.bindings: Dict[str, Any] = {}
        self.output_futures: List[Future] = []

        # Set positional arguments
        for i, arg_value in enumerate(args):
            if i >= len(self.func.inputs):
                raise ValueError(
                    f"Function {self.func.name} got too many positional arguments."
                )
            self._set_value(self.func.inputs[i], arg_value, self.bindings)

        # Set keyword arguments
        for name, arg_value in kwargs.items():
            assert (
                name not in self.bindings
            ), f"Function {self.func.name} got multiple values for argument {name}"
            assert (
                name in self.func.params_map
            ), f"Function {self.func.name} got an unexpected keyword argument {name}"
            param = self.func.params_map[name]
            if param in self.func.outputs:
                raise ValueError(
                    f"Argument {name} is an output parameter hence cannot be set."
                )
            self._set_value(param, arg_value, self.bindings)

        # Create output futures
        for param in self.func.outputs:
            future = Future()
            self.output_futures.append(future)
            self._set_value(param, future, self.bindings)

    @staticmethod
    def _set_value(param: Parameter, value: Any, bindings: Dict[str, Any]):
        if param.typ != ParamType.INPUT_PYOBJ:
            if not isinstance(value, str) and not isinstance(value, Future):
                raise TypeError(
                    f"Argument {param.name} in an input loc should be a str or a Future, "
                    f"but got {type(value)}: {value}"
                )
        else:
            # For Python object, we use __str__ instead of __repr__ to serialize it.
            value = str(value)
        bindings[param.name] = value

    # NOTE(chaofan): We use pickle to serialize the call.
    # We use protocol=0 to make the result can be passed by http.
    # There maybe some better ways to do this, but this is not important for this project.

    def pickle(self) -> str:
        return str(pickle.dumps(self, protocol=0), encoding="ascii")

    @classmethod
    def unpickle(cls, pickled: str) -> "SemanticCall":
        return pickle.loads(bytes(pickled, encoding="ascii"))
