# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import types
from abc import abstractmethod, ABC
import marshal
from typing import List, Dict, Type, Optional, Any, Set, Union, Tuple, Callable
import regex as re
from dataclasses import dataclass

from parrot.utils import get_logger

from .semantic_variable import (
    SemanticVariable,
    SemanticRegion,
    Constant,
    Parameter,
    ParamType,
    ParameterLoc,
)

from .function_call import SemanticCall, NativeCall


logger = get_logger("Function")


@dataclass
class NativeFuncMetadata:
    """Metadata of a native function."""

    timeout: float


@dataclass
class SemaFuncMetadata:
    """Metadata of a semantic function."""

    cache_prefix: bool
    remove_pure_fill: bool
    models: List[str]


class BasicFunction(ABC):
    """Basic class of functions."""

    _virtual_machine_env: Optional["VirtualMachine"] = None

    def __init__(self, name: str, params: List[Parameter]):
        self.name = name
        self.params = params
        self.params_map = dict([(param.name, param) for param in self.params])
        self.inputs = [
            param for param in self.params if param.typ != ParamType.OUTPUT_LOC
        ]
        self.outputs = [
            param for param in self.params if param.typ == ParamType.OUTPUT_LOC
        ]


class NativeFunction(BasicFunction):
    """A native function.

    It should be defined by a Python function, with inputs and outputs as strings.
    """

    def __init__(self, name, pyfunc: Callable, params: list[Parameter], **kwargs):
        super().__init__(name, params)

        self.pyfunc_code_dumped = marshal.dumps(pyfunc.__code__)
        self.metadata = NativeFuncMetadata(**kwargs)

        if BasicFunction._virtual_machine_env is not None:
            BasicFunction._virtual_machine_env.register_function_handler(self)

    def __call__(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "SemanticCall"]:
        """Call to a native function."""

        return self._call_func(*args, **kwargs)

    def invoke(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "SemanticCall"]:
        """Same as __call__."""

        return self._call_func(*args, **kwargs)

    async def ainvoke(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "SemanticCall"]:
        """Async call."""

        return await self._acall_func(*args, **kwargs)

    def _call_func(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "SemanticCall"]:
        call = NativeCall(self, *args, **kwargs)
        if BasicFunction._virtual_machine_env is not None:
            BasicFunction._virtual_machine_env.submit_call_handler(call)
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return call

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

    async def _acall_func(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "SemanticCall"]:
        call = NativeCall(self, *args, **kwargs)
        if BasicFunction._virtual_machine_env is not None:
            # Different from _call_func, we use asubmit_call_handler here.
            await BasicFunction._virtual_machine_env.asubmit_call_handler(call)
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return call

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

    def display_signature(self) -> str:
        """Display the function signature."""
        return f"{self.name}({', '.join([f'{param.name}: {param.typ}' for param in self.params])})"

    def get_pyfunc(self) -> Callable:
        if self.name in globals():
            return globals()[self.name]
        code_decoded = marshal.loads(self.pyfunc_code_dumped)
        return types.FunctionType(code_decoded, globals(), self.name)


def push_to_body(piece_cls: Type[SemanticRegion], body: List[SemanticRegion], **kwargs):
    idx = len(body)
    body.append(piece_cls(idx=idx, **kwargs))


def parse_func_body(
    body_str: str,
    params_map: Dict[str, Parameter],
    metadata: SemaFuncMetadata,
) -> List[SemanticRegion]:
    """Parse the function body string to a list of semantic variables."""

    PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]*}}"
    pattern = re.compile(PLACEHOLDER_REGEX)
    iterator = pattern.finditer(body_str)
    last_pos = 0

    ret: List[SemanticRegion] = []

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
                isinstance(ret[-1], ParameterLoc) and ret[-1].param.is_output
            ), "Output loc can't be adjacent to another output loc."
            assert not param.name in outputs, "Output param can't be used twice."
            outputs.add(param.name)
        push_to_body(ParameterLoc, ret, param=param)

        if param.is_output:
            last_output_loc_idx = len(ret) - 1

        last_pos = match.end()

    if metadata.remove_pure_fill:
        # NOTE(chaofan): we prune all pieces after the last output loc.
        # The following code is also correct for last_output_loc_idx == -1.
        ret = ret[: last_output_loc_idx + 1]
    elif last_pos < len(body_str):
        push_to_body(Constant, ret, text=body_str[last_pos:])

    return ret


@dataclass
class SemanticFunction(BasicFunction):
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
        func_body: Optional[List[SemanticRegion]] = None,
        **kwargs,
    ):
        """For semantic function, function body is just a prompt template.
        After parsed, it turns to be a list of semantic variables.
        """

        # ---------- Basic Info ----------
        super().__init__(name, params)
        self.metadata = SemaFuncMetadata(**kwargs)
        if func_body_str is not None:
            self.body: List[SemanticRegion] = parse_func_body(
                func_body_str, self.params_map, self.metadata
            )
        elif func_body is not None:
            self.body = func_body
        else:
            raise ValueError("Either func_body_str or func_body should be provided.")

        if BasicFunction._virtual_machine_env is not None:
            BasicFunction._virtual_machine_env.register_function_handler(self)

    def __call__(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], SemanticCall]:
        """Call to a semantic function.

        Some NOTES:

        - Calling a parrot semantic function will not execute it immediately.
          Instead, this will submit the call to OS.

        - The return value is a list of SemanticVariable objects, which can be used to get the
          output contents or passed to other functions.

        - When passing arguments, the caller needs to pass all the input arguments, including
          INPUT_LOC and INPUT_PYOBJ.

        - In some cases, the caller may preallocate the output SemanticVariables. In this case, the caller
          can also pass them as arguments to the function, to make the outputs be written to
          the preallocated SemanticVariables. But in order to make the call convention clear, we only
          allow these arguments to be passed as keyword arguments.

        - The INPUT_PYOBJ arguments should be Python objects, which will be turns to a string
          using __str__ method.
        """

        return self._call_func(None, *args, **kwargs)

    def invoke(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], SemanticCall]:
        """Same as __call__."""

        return self._call_func(None, *args, **kwargs)

    async def ainvoke(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], SemanticCall]:
        """Async call."""

        return await self._acall_func(None, *args, **kwargs)

    def invoke_statefully(
        self,
        context_successor: "SemanticFunction",
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], SemanticCall]:
        """Call a semantic function statefully.

        This means the context of the function will not be freed immediately. Instead, it will
        be passed to the successor function.

        For example, if we execute `f1.invoke_statefully(f2, ...)`, then the next call to `f2`
        will use the context of `f1` this round.
        """

        return self._call_func(context_successor, *args, **kwargs)

    def _call_func(
        self,
        context_successor: Optional["SemanticFunction"],
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], SemanticCall]:
        call = SemanticCall(self, context_successor, *args, **kwargs)
        if BasicFunction._virtual_machine_env is not None:
            BasicFunction._virtual_machine_env.submit_call_handler(call)
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return call

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

    async def _acall_func(
        self,
        context_successor: Optional["SemanticFunction"],
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], SemanticCall]:
        call = SemanticCall(self, context_successor, *args, **kwargs)
        if BasicFunction._virtual_machine_env is not None:
            # Different from _call_func, we use asubmit_call_handler here.
            await BasicFunction._virtual_machine_env.asubmit_call_handler(call)
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return call

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

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
