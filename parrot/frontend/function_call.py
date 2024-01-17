# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from abc import abstractmethod, ABC

import pickle
import base64
from typing import List, Any, Dict, Optional, Callable

from .semantic_variable import Parameter, ParamType, SemanticVariable


class BasicCall(ABC):
    """Basic call model."""

    def __init__(
        self, func: "BasicFunction", *args: List[Any], **kwargs: Dict[str, Any]
    ):
        # ---------- Basic Info ----------
        self.func = func
        self.bindings: Dict[str, Any] = {}
        self.output_vars: List[Any] = []

        # ---------- Runtime ----------
        self.edges: List["DAGEdge"] = []
        # idx of region -> edge. For native call, always 0.
        self.edges_map: Dict[int, "DAGEdge"] = {}

        self.set_bindings(args, kwargs)

    @abstractmethod
    def pickle(self) -> str:
        """Pickle the call."""

    @classmethod
    @abstractmethod
    def unpickle(cls, pickled: str) -> "BasicCall":
        """Unpickle the call."""

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
            self._set_value(param, arg_value, self.bindings)

        # Create output variables.
        for param in self.func.outputs:
            # Skip the output locs that are already set.
            if param.name not in self.bindings:
                out_var = SemanticVariable(name=param.name)
                self.output_vars.append(out_var)
                self._set_value(param, out_var, self.bindings)

    @staticmethod
    def _set_value(param: Parameter, value: Any, bindings: Dict[str, Any]):
        if param.typ != ParamType.INPUT_PYOBJ:
            if not isinstance(value, str) and not isinstance(value, SemanticVariable):
                raise TypeError(
                    f"Argument {param.name} in an input loc should be a str or a SemanticVariable, "
                    f"but got {type(value)}: {value}"
                )
        else:
            # For Python object, we use __str__ instead of __repr__ to serialize it.
            value = str(value)
        bindings[param.name] = value


class SemanticCall(BasicCall):
    """A call to a semantic function."""

    def __init__(
        self,
        func: "SemanticFunction",
        context_successor: Optional["SemanticFunction"],
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ):
        # ---------- Basic Info ----------
        super().__init__(func, *args, **kwargs)
        self.context_successor: Optional[str] = (
            context_successor.name if context_successor else None
        )

        # ---------- Runtime ----------
        self.thread: Optional["Thread"] = None

    # NOTE(chaofan): We use pickle to serialize the call.
    # We use protocol=0 to make the result can be passed by http.
    # There maybe some better ways to do this, but this is not important for this project.

    def pickle(self) -> str:
        dumped = pickle.dumps(self, protocol=0)  # binary codes
        b64str = str(base64.b64encode(dumped), encoding="ascii")  # base64 string
        return b64str

    @classmethod
    def unpickle(cls, pickled: str) -> "SemanticCall":
        pickled = bytes(pickled, encoding="ascii")
        b64bytes = base64.b64decode(pickled)
        call = pickle.loads(b64bytes)
        return call


class NativeCall(BasicCall):
    """A call to a native function."""

    def __init__(
        self,
        func: "NativeFunction",
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ):
        # ---------- Basic Info ----------
        super().__init__(func, *args, **kwargs)

    # NOTE(chaofan): We use pickle to serialize the call.
    # We use protocol=0 to make the result can be passed by http.
    # There maybe some better ways to do this, but this is not important for this project.

    def pickle(self) -> str:
        dumped = pickle.dumps(self, protocol=0)  # binary codes
        b64str = str(base64.b64encode(dumped), encoding="ascii")  # base64 string
        return b64str

    @classmethod
    def unpickle(cls, pickled: str) -> "SemanticCall":
        pickled = bytes(pickled, encoding="ascii")
        b64bytes = base64.b64decode(pickled)
        call = pickle.loads(b64bytes)
        return call
