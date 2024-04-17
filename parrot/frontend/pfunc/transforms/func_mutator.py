# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict, List
from abc import ABC, abstractmethod
from dataclasses import asdict
from ..function import (
    SemanticFunction,
    FuncBodyPiece,
    ParameterLoc,
    Constant,
    Parameter,
)


class FuncMutator(ABC):
    """Abstract class for function mutators."""

    def transform(self, func: SemanticFunction) -> SemanticFunction:
        """Transform a function and return a new function."""

        new_params: List[Parameter] = []
        new_body: List[FuncBodyPiece] = []

        self._param_remap: Dict[str, Parameter] = {}
        for param in func.params:
            new_param = self._visit_parameter(param)
            self._param_remap[param.name] = new_param
            new_params.append(new_param)

        for piece in func.body:
            new_body.append(self._visit_body_piece(piece))

        new_func = SemanticFunction(
            name=func.name,
            params=new_params,
            func_body=new_body,
            try_register=False,
            **asdict(func.metadata),
        )

        return self._visit_func(new_func)

    @abstractmethod
    def _visit_func(self, func: SemanticFunction) -> SemanticFunction:
        """Visit a function and return a new function.

        NOTE(chaofan): This method is only used to mutate the basic info of the function, like
        name, type, the order of the body pieces. The body pieces themselves are mutated by
        `_visit_body_piece`.
        """

        raise NotImplementedError

    def _visit_body_piece(self, body_piece: FuncBodyPiece) -> FuncBodyPiece:
        """Visit a function body piece and return a new function body piece.

        NOTE(chaofan): We don't change to idx of the body pieces here. We only focus on mutating
        the pieces themselves.
        """

        if isinstance(body_piece, Constant):
            return self._visit_constant(body_piece)
        elif isinstance(body_piece, ParameterLoc):
            return self._visit_param_loc(body_piece)
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
