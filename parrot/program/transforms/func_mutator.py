# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List
from abc import ABC, abstractmethod
from dataclasses import asdict
from ..function import (
    SemanticFunction,
    SemanticVariable,
    ParameterLoc,
    Constant,
    Parameter,
)


class FuncMutator(ABC):
    """Abstract class for function mutators."""

    def transform(self, func: SemanticFunction) -> SemanticFunction:
        """Transform a function and return a new function."""

        new_params: List[Parameter] = []
        new_body: List[SemanticVariable] = []

        self._param_remap: Dict[str, Parameter] = {}
        for param in func.params:
            new_param = self._visit_parameter(param)
            self._param_remap[param.name] = new_param
            new_params.append(new_param)

        for piece in func.body:
            new_body.append(self._visit_sv(piece))

        new_func = SemanticFunction(
            name=func.name,
            params=new_params,
            func_body=new_body,
            **asdict(func.metadata),
        )

        return self._visit_func(new_func)

    @abstractmethod
    def _visit_func(self, func: SemanticFunction) -> SemanticFunction:
        """Visit a function and return a new function.

        NOTE(chaofan): This method is only used to mutate the basic info of the function, like
        name, type, the order of the pieces. The pieces themselves are mutated by `visit_func_piece`.
        """

        raise NotImplementedError

    def _visit_sv(self, sv: SemanticVariable) -> SemanticVariable:
        """Visit a semantic variable and return a new semantic variable.

        NOTE(chaofan): We don't change to idx of the pieces here. We only focus on mutating
        the pieces themselves.
        """

        if isinstance(sv, Constant):
            return self._visit_constant(sv)
        elif isinstance(sv, ParameterLoc):
            return self._visit_param_loc(sv)
        else:
            raise NotImplementedError

    @abstractmethod
    def _visit_constant(self, constant: Constant) -> Constant:
        raise NotImplementedError

    def _visit_param_loc(self, param_loc: ParameterLoc) -> ParameterLoc:
        return ParameterLoc(
            idx=param_loc.idx, param=self._param_remap[param_loc.param.name]
        )

    @abstractmethod
    def _visit_parameter(self, param: Parameter) -> Parameter:
        raise NotImplementedError
