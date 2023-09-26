from typing import Dict, List
from abc import ABC, abstractmethod
from ..function import (
    SemanticFunction,
    FunctionPiece,
    ParameterLoc,
    Constant,
    Prefix,
    Parameter,
)


class FuncMutator(ABC):
    """Abstract class for function mutators."""

    def transform(self, func: SemanticFunction) -> SemanticFunction:
        """Transform a function and return a new function."""

        new_params: List[Parameter] = []
        new_body: List[FunctionPiece] = []

        self._param_remap: Dict[str, Parameter] = {}
        for param in func.params:
            new_param = self._visit_parameter(param)
            self._param_remap[param.name] = new_param
            new_params.append(new_param)

        for piece in func.body:
            new_body.append(self._visit_func_piece(piece))

        new_func = SemanticFunction(
            name=func.name,
            params=new_params,
            cached_prefix=func.cached_prefix,
            func_body=new_body,
        )

        return self._visit_func(new_func)

    @abstractmethod
    def _visit_func(self, func: SemanticFunction) -> SemanticFunction:
        """Visit a function and return a new function.

        NOTE(chaofan): This method is only used to mutate the basic info of the function, like
        name, type, the order of the pieces. The pieces themselves are mutated by `visit_func_piece`.
        """

        raise NotImplementedError

    def _visit_func_piece(self, func_piece: FunctionPiece) -> FunctionPiece:
        """Visit a function piece and return a new function piece.

        NOTE(chaofan): We don't change to idx of the pieces here. We only focus on mutating
        the pieces themselves.
        """

        if isinstance(func_piece, Prefix):
            return self._visit_prefix(func_piece)
        elif isinstance(func_piece, Constant):
            return self._visit_constant(func_piece)
        elif isinstance(func_piece, ParameterLoc):
            return self._visit_param_loc(func_piece)
        else:
            raise NotImplementedError

    @abstractmethod
    def _visit_prefix(self, prefix: Prefix) -> Prefix:
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
