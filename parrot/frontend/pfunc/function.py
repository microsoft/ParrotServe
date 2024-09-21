# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import marshal
from abc import ABC
import types
from typing import Tuple, Callable, List, Dict, Type, Optional, Any, Set, Union
import regex as re
from dataclasses import dataclass, asdict

from parrot.utils import (
    get_logger,
    serialize_func_code,
    deserialize_func_code,
    bytes_to_encoded_b64str,
)

from parrot.serve.graph.call_request import (
    SemanticCallMetadata as SemanticFuncMetadata,
    NativeCallMetadata as NativeFuncMetadata,
)

from .function_body import FuncBodyPiece, Constant, Parameter, ParamType, ParameterLoc
from .semantic_variable import SemanticVariable


logger = get_logger("PFunc Function")


# ---------- Basic ----------


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

    # ---------- VM Env Methods ----------

    def _has_vm_env(self) -> bool:
        return BasicFunction._virtual_machine_env is not None

    def _register_function(self) -> None:
        if self._has_vm_env():
            BasicFunction._virtual_machine_env.register_function_handler(self)
        else:
            logger.warning(
                f'VM environment is not set. Not register the function "{self.name}".'
            )


class BasicCall(ABC):
    """Basic call model."""

    def __init__(
        self, func: "BasicFunction", *args: List[Any], **kwargs: Dict[str, Any]
    ):
        # ---------- Basic Info ----------
        self.func = func
        self.bindings: Dict[str, Any] = {}
        self.output_vars: List[Any] = []

        self.set_bindings(args, kwargs)

    def set_bindings(
        self,
        args: List[Any],
        kwargs: Dict[str, Any],
    ):
        """Set the bindings for the call."""

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
            # if param in self.func.outputs:
            #     raise ValueError(
            #         f"Argument {name} is an output parameter hence cannot be set."
            #     )
            # Can only be keyword arguments for Outputs.
            self._set_value(param, arg_value, self.bindings)

        # Create output variables.
        for param in self.func.outputs:
            # Skip the output locs that are already set.
            if param.name not in self.bindings:
                out_var = SemanticVariable(name=param.name, register=False)
                self.output_vars.append(out_var)
                self._set_value(param, out_var, self.bindings)

    @staticmethod
    def _set_value(param: Parameter, value: Any, bindings: Dict[str, Any]) -> None:
        # Python object
        if param.typ == ParamType.INPUT_PYOBJ:
            bindings[param.name] = value
        elif param.typ == ParamType.INPUT_LOC:
            if not isinstance(value, SemanticVariable):
                # For Python object, we use __str__ instead of __repr__ to serialize it.
                value = str(value)

                # Create a new SemanticVariable for this type of input.
                in_var = SemanticVariable(name=param.name, register=False)
                # in_var.set(value)
                in_var.content = value  # It has initial value.
                bindings[param.name] = in_var
            else:
                # Referring existing SemanticVariable.
                bindings[param.name] = value
        else:
            if not isinstance(value, SemanticVariable):
                raise TypeError(
                    f"Value of argument {param.name} in an output loc should be a SemanticVariable, "
                    f"but got {type(value)}: {value}"
                )
            bindings[param.name] = value

    def update_var_ids(self, param_info: List[Dict]) -> None:
        for mapping in param_info:
            param_name = mapping["parameter_name"]
            var_id = mapping["var_id"]
            var = self.bindings[param_name]
            assert isinstance(var, SemanticVariable), f"Unexpected var type: {var}"
            var.assign_id(var_id)


# ---------- Semantic Function ----------


# Move to serve/graph/call_request.py
# @dataclass
# class SemanticFuncMetadata:
#     """Metadata of a semantic function."""

#     remove_pure_fill: bool
#     models: List[str]
#     model_type: str


@dataclass
class ParameterLoc(FuncBodyPiece):
    """An input/output location in the function."""

    param: Parameter


def push_to_body(piece_cls: Type[FuncBodyPiece], body: List[FuncBodyPiece], **kwargs):
    idx = len(body)
    body.append(piece_cls(idx=idx, **kwargs))


def parse_func_body(
    body_str: str,
    params_map: Dict[str, Parameter],
    metadata: SemanticFuncMetadata,
) -> List[FuncBodyPiece]:
    """Parse the function body string to a list of semantic variables."""

    PLACEHOLDER_REGEX = "{{[a-zA-Z_][a-zA-Z0-9_]*}}"
    pattern = re.compile(PLACEHOLDER_REGEX)
    iterator = pattern.finditer(body_str)
    last_pos = 0

    ret: List[FuncBodyPiece] = []

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
        func_body: Optional[List[FuncBodyPiece]] = None,
        try_register: bool = True,
        **metadata_kwargs,
    ):
        """For semantic function, function body is just a prompt template.
        After parsed, it turns to be a list of semantic variables.
        """

        # ---------- Basic Info ----------
        super().__init__(name, params)
        metadata_dict = SemanticFuncMetadata.get_default_dict()
        metadata_dict.update(**metadata_kwargs)
        self.metadata = SemanticFuncMetadata(**metadata_dict)

        if func_body_str is not None:
            self.body: List[FuncBodyPiece] = parse_func_body(
                func_body_str, self.params_map, self.metadata
            )
        elif func_body is not None:
            self.body = func_body
        else:
            raise ValueError("Either func_body_str or func_body should be provided.")

        if try_register:
            # This will generate a register warning if the VM environment is not set.
            self._register_function()

    # ---------- VM Env Methods ----------

    def _submit_semantic_call(self, call: "SemanticCall") -> List:
        if self._has_vm_env():
            return BasicFunction._virtual_machine_env.submit_semantic_call_handler(call)
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return []

    async def _asubmit_semantic_call(self, call: "SemanticCall") -> List:
        if self._has_vm_env():
            return (
                await BasicFunction._virtual_machine_env.asubmit_semantic_call_handler(
                    call
                )
            )
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return []

    # ---------- Call Methods ----------

    def __call__(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...]]:
        """Call to a semantic function.

        Some NOTES:

        - Calling a parrot semantic function will not execute it immediately.
          Instead, this will submit the call to ServeLayer.

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
        call = SemanticCall(self, *args, **kwargs)

        param_info = self._submit_semantic_call(call)
        if not self._has_vm_env():
            return call
        else:
            call.update_var_ids(param_info)

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

    async def _acall_func(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "SemanticCall"]:
        call = SemanticCall(self, *args, **kwargs)

        param_info = await self._asubmit_semantic_call(call)
        if not self._has_vm_env():
            return call
        else:
            call.update_var_ids(param_info)

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

    def to_template_str(self) -> str:
        """Convert the function body to template string."""

        return "".join(
            [
                (
                    piece.text
                    if isinstance(piece, Constant)
                    else piece.param.get_param_str()
                )
                for piece in self.body
            ]
        )


class SemanticCall(BasicCall):
    """A call to a semantic function."""

    def __init__(
        self,
        func: "SemanticFunction",
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ):
        super().__init__(func, *args, **kwargs)

    def to_request_payload(self) -> Dict:
        """Convert the call to a request payload."""

        payload = asdict(self.func.metadata)
        template_str: str = self.func.to_template_str()
        parameters = []

        for param in self.func.params:
            param_value = self.bindings[param.name]

            param_dict = {
                "name": param.name,
                "is_output": param.is_output,
            }

            if param.is_output:
                assert (
                    param.sampling_config is not None
                ), "Output loc must have sampling config."

                param_dict["sampling_config"] = asdict(param.sampling_config)

            if isinstance(param_value, SemanticVariable):
                # For P.Input or P.Output
                if param_value.is_registered:
                    param_dict["var_id"] = param_value.id
                elif not param.is_output:  # For input, add initial value.
                    param_dict["value"] = param_value.get()
                parameters.append(param_dict)
            else:
                # Directly render the Python object to the template string.
                # For Python object, we use __str__ instead of __repr__ to serialize it.
                param_value = str(param_value)
                param_str = param.get_param_str()
                template_str = template_str.replace(
                    param_str, param_value
                )  # Render the template string

            # else:
            #     raise ValueError(f"Unexpected param value: {param_value}")

        payload["template"] = template_str
        payload["parameters"] = parameters
        payload["func_name"] = self.func.name

        return payload


# ---------- Python Native Function ----------


class PyNativeFunction(BasicFunction):
    """Python native function.

    It should be defined by a Python function, with inputs and outputs as strings.
    """

    def __init__(
        self,
        name,
        pyfunc: Callable,
        params: List[Parameter],
        try_register: bool = True,
        **metadata_kwargs,
    ):
        super().__init__(name, params)

        self.pyfunc_code_dumped = serialize_func_code(pyfunc.__code__)
        metadata_dict = NativeFuncMetadata.get_default_dict()
        metadata_dict.update(**metadata_kwargs)
        self.metadata = NativeFuncMetadata(**metadata_dict)

        if try_register:
            # This will generate a register warning if the VM environment is not set.
            self._register_function()

    # ---------- VM Env Methods ----------

    def _submit_native_call(self, call: "PyNativeCall") -> List:
        if self._has_vm_env():
            return BasicFunction._virtual_machine_env.submit_py_native_call_handler(
                call
            )
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return []

    async def _asubmit_native_call(self, call: "PyNativeCall") -> List:
        if self._has_vm_env():
            return (
                await BasicFunction._virtual_machine_env.asubmit_py_native_call_handler(
                    call
                )
            )
        else:
            logger.warning(
                "VM environment is not set. Not submit the Call. Return Call instead. "
                "(Please run a Parrot function under a VM context.)"
            )
            return []

    def __call__(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "PyNativeCall"]:
        """Call to a native function."""

        return self._call_func(*args, **kwargs)

    def invoke(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "PyNativeCall"]:
        """Same as __call__."""

        return self._call_func(*args, **kwargs)

    async def ainvoke(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "PyNativeCall"]:
        """Async call."""

        return await self._acall_func(*args, **kwargs)

    def _call_func(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "PyNativeCall"]:
        call = PyNativeCall(self, *args, **kwargs)

        param_info = self._submit_native_call(call)
        if not self._has_vm_env():
            return call
        else:
            call.update_var_ids(param_info)

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

    async def _acall_func(
        self,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Union[SemanticVariable, Tuple[SemanticVariable, ...], "PyNativeCall"]:
        call = PyNativeCall(self, *args, **kwargs)

        param_info = self._submit_native_call(call)
        if not self._has_vm_env():
            return call
        else:
            call.update_var_ids(param_info)

        # Unpack the output SemanticVariables
        if len(call.output_vars) == 1:
            return call.output_vars[0]
        return tuple(call.output_vars)

    def display_signature(self) -> str:
        """Display the function signature."""
        return f"{self.name}({', '.join([f'{param.name}: {param.typ}' for param in self.params])})"

    def get_pyfunc(self) -> Callable:
        code_deserialized = deserialize_func_code(self.pyfunc_code_dumped)

        # Here for the scope Dict, we pass {} because we don't want to pollute the scope.
        # Hence the pyfunc we get is just a temporary one.
        return types.FunctionType(code_deserialized, {}, self.name)


class PyNativeCall(BasicCall):
    """A call to a native function."""

    def __init__(
        self,
        func: "PyNativeFunction",
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ):
        # ---------- Basic Info ----------
        super().__init__(func, *args, **kwargs)

    def to_request_payload(self, with_code: bool = True) -> Dict:
        """Convert the call to a request payload."""

        payload = asdict(self.func.metadata)

        parameters = []

        for param in self.func.params:
            param_value = self.bindings[param.name]

            param_dict = {
                "name": param.name,
                "is_output": param.is_output,
            }

            if isinstance(param_value, SemanticVariable):
                if param_value.is_registered:
                    param_dict["var_id"] = param_value.id
                elif not param.is_output:  # For input, add initial value.
                    param_dict["value"] = param_value.get()
            else:
                param_dict["value"] = param_value

            parameters.append(param_dict)

        payload["parameters"] = parameters
        payload["func_name"] = self.func.name

        if with_code:
            payload["func_code"] = bytes_to_encoded_b64str(self.func.pyfunc_code_dumped)

        return payload
