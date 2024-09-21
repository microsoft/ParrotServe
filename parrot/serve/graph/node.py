# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import asyncio
from typing import List, Optional, Dict, Any, Union, Type

from parrot.sampling_config import SamplingConfig
from parrot.exceptions import parrot_assert

from .call_request import SemanticFunctionParameter, PyNativeCallRequest
from .semantic_variable import SemanticVariable


class BaseNode:
    """Represent a computational node in the graph."""

    def __init__(self):
        self._id_in_graph: Optional[int] = None

    # ---------- Graph ----------

    @property
    def is_inserted(self) -> bool:
        """Whether the node is inserted into the graph."""

        return self._id_in_graph is not None

    @property
    def id_in_graph(self) -> int:
        parrot_assert(self.is_inserted, "This node is not inserted.")
        return self._id_in_graph

    def set_id_in_graph(self, id_in_graph: int) -> None:
        self._id_in_graph = id_in_graph

    # ---------- Polling ----------

    async def wait_ready(self) -> None:
        """Wait until the node is ready."""

        raise NotImplementedError

    # ---------- Display ----------

    def _get_display_elements(self) -> Dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join([f"{k}={v}" for k, v in self._get_display_elements().items()])
            + ")"
        )

    def pretty_print(self) -> str:
        """Pretty print the node."""

        ret = self.__class__.__name__ + ":\n"
        for k, v in self._get_display_elements().items():
            ret += f"\t{k}: {v}\n"
        return ret

    def _short_repr_add_graph_id(self, repr: str) -> str:
        if self.is_inserted:
            return f"{self._id_in_graph}: " + repr
        return repr

    def short_repr(self) -> str:
        """Short representation of the node."""

        return self._short_repr_add_graph_id("BaseNode")


class SemanticNode(BaseNode):
    """Represent a semantic computational node in the graph."""

    def __init__(self):
        super().__init__()

        self._sv: Optional[SemanticVariable] = None
        self._completion_chain: Optional["CompletionChain"] = None
        # self.request_chain: Optional["RequestChain"] = None

        # Edge type A: Fill -> Fill -> Fill -> Gen -> Fill -> Fill -> Gen -> ...
        self._edge_a_prev_node: Optional["SemanticNode"] = None
        self._edge_a_next_node: Optional["SemanticNode"] = None

    @property
    def is_gen(self) -> bool:
        """Whether the node is a Gen node."""

        return isinstance(self, PlaceholderGen)

    @property
    def has_placeholder(self) -> bool:
        """Whether the node has a placeholder."""

        return not isinstance(self, ConstantFill)

    # ---------- SV ----------

    @property
    def has_sv(self) -> bool:
        """Whether the node has a SV."""

        return self._sv is not None

    @property
    def sv(self) -> SemanticVariable:
        parrot_assert(self.has_sv, "This node has no SV.")
        return self._sv

    def set_sv(self, sv: SemanticVariable) -> None:
        self._sv = sv

    @property
    def sv_name(self) -> str:
        # parrot_assert(self.has_var, "This node has no SV.")
        if not self.has_sv:
            return "(no SV)"
        return self.sv.name

    @property
    def var_id(self) -> str:
        # parrot_assert(self.has_var, "This node has no SV.")
        if not self.has_sv:
            return "(no SV)"
        return self.sv.id

    def get(self) -> str:
        """Get the content of the node."""

        return self.sv.get()

    # ---------- Graph ----------

    def link_edge_a_with(self, prev_node: "SemanticNode") -> None:
        """Link the node with its predecessor in edge type A."""

        self._edge_a_prev_node = prev_node
        prev_node._edge_a_next_node = self

    @property
    def has_edge_a_prev_node(self) -> bool:
        return self._edge_a_prev_node is not None

    @property
    def has_edge_a_next_node(self) -> bool:
        return self._edge_a_next_node is not None

    def get_edge_a_prev_node(self) -> Optional["SemanticNode"]:
        return self._edge_a_prev_node

    def get_edge_a_next_node(self) -> Optional["SemanticNode"]:
        return self._edge_a_next_node

    @property
    def has_edge_b_prev_node(self) -> bool:
        return self.sv.has_producer

    def get_edge_b_prev_node(self) -> Optional["SemanticNode"]:
        """Edge type B: prev node. 0 or 1."""

        parrot_assert(self.is_inserted, "Should be inserted before get DAG info.")
        return self.sv.get_producer()

    def get_edge_b_next_nodes(self) -> List["SemanticNode"]:
        """Edge type B: next node. Only Gen node has multiple next nodes."""

        parrot_assert(self.is_inserted, "Should be inserted before get DAG info.")
        if not isinstance(self, PlaceholderGen):
            return []
        return self.sv.get_consumers()

    @property
    def comp_chain(self) -> "CompletionChain":
        parrot_assert(self.is_inserted, "Should be inserted before get DAG info.")
        parrot_assert(self.comp_chain_is_set, "This node has no completion chain.")
        return self._completion_chain

    @property
    def comp_chain_is_set(self) -> bool:
        return self._completion_chain is not None

    def set_comp_chain(self, comp_chain: "CompletionChain") -> None:
        self._completion_chain = comp_chain

    # ---------- Polling ----------

    async def wait_ready(self) -> None:
        """Wait until the node is ready. A node is ready if all its inputs are ready
        and the SV is ready.

        To be specific, a node in our graph can only have at most 2 inputs:
        - Predecessor in edge type A (previous Fill)
        - Predecessor in edge type B (Gen in the same SV)

        The node is ready iff. all its predecessors' SVs are ready.
        """

        if self._edge_a_prev_node is not None:
            await self._edge_a_prev_node.sv.wait_ready()

        await self.sv.wait_ready()


class ConstantFill(SemanticNode):
    """Represent a fill node (constant) in the graph."""

    def __init__(self, constant_text: str):
        super().__init__()
        self.constant_text = constant_text

    def _get_display_elements(self) -> Dict:
        return {
            "sv_name": self.sv_name,
            "var_id": self.var_id,
            "constant_text": self.constant_text,
        }

    def short_repr(self) -> str:
        length_threshold = 7
        short_text = (
            self.constant_text[:length_threshold] + "..."
            if len(self.constant_text) > length_threshold
            else self.constant_text
        )
        return self._short_repr_add_graph_id("ConstantFill(" + short_text + ")")


class PlaceholderFill(SemanticNode):
    """Represent a fill node (placeholder) in the graph."""

    def __init__(self, parameter: SemanticFunctionParameter):
        super().__init__()
        self.placeholder_param = parameter  # The referred parameter of this placeholder

    def _get_display_elements(self) -> Dict:
        return {
            "sv_name": self.sv_name,
            "var_id": self.var_id,
            "parameter_name": self.placeholder_param.name,
        }

    def short_repr(self) -> str:
        return self._short_repr_add_graph_id(
            f"PlaceholderFill({self.placeholder_param.name})"
        )


class PlaceholderGen(SemanticNode):
    """Represent a gen node (placeholder, actually it must be) in the graph."""

    def __init__(self, parameter: SemanticFunctionParameter):
        super().__init__()
        self.placeholder_param = parameter  # The referred parameter of this placeholder

    @property
    def sampling_config(self) -> SamplingConfig:
        return self.placeholder_param.sampling_config

    def _get_display_elements(self) -> Dict:
        return {
            "sv_name": self.sv_name,
            "var_id": self.var_id,
            "parameter_name": self.placeholder_param.name,
            "sampling_config": self.sampling_config,
        }

    def short_repr(self) -> str:
        return self._short_repr_add_graph_id(
            f"PlaceholderGen({self.placeholder_param.name})"
        )

    # async def wait_ready(self):
    #     """NOTE(chaofan): We don't need to wait Gen to be ready."""
    #     pass


class NativeFuncNode(BaseNode):
    """Represent a native function in the graph."

    For simplicity, we represent the entire native function as a big node in the graph.
    """

    def __init__(self, native_func: PyNativeCallRequest):
        super().__init__()

        self.native_func = native_func
        self.input_vars: Dict[str, SemanticVariable] = {}
        self.input_values: Dict[str, Any] = {}
        self.output_vars: Dict[str, SemanticVariable] = {}

        # Only valid after inserted into a graph.
        self._param_info: List[Dict] = []

        # Add values
        for key, param in native_func.parameters_map.items():
            if param.has_value:
                self.input_values[key] = param.value

    def get_param_info(self) -> List[Dict]:
        """Get the param info after inserted into a graph.

        Returns:
            List[Dict]: param info.
        """

        parrot_assert(
            self.is_inserted,
            "Get param info failed: RequestChain has not been inserted into a graph.",
        )
        return self._param_info

    def get_prev_producers(self) -> List[BaseNode]:
        """Get the previous producers of this node."""

        prev_producers = []
        for sv in self.input_vars.values():
            producer = sv.get_producer()
            if producer is not None:
                prev_producers.append(producer)
        return prev_producers

    def get_next_consumers(self) -> List[BaseNode]:
        """Get the next consumers of this node."""

        next_consumers = []
        for sv in self.output_vars.values():
            consumers = sv.get_consumers()
            next_consumers.extend(consumers)
        return next_consumers

    def _get_display_elements(self) -> Dict:
        return {
            "func_name": self.native_func.func_name,
        }

    def short_repr(self) -> str:
        return self._short_repr_add_graph_id(
            f"NativeFuncNode({self.native_func.func_name})"
        )

    @classmethod
    def from_variables(
        cls,
        func_name: str,
        input_vars: Dict[str, SemanticVariable],
        output_vars: Dict[str, SemanticVariable],
    ) -> "NativeFuncNode":
        """Create a NativeFuncNode from input and output SVs.

        Only for displaying.
        """

        pseudo_native_func = PyNativeCallRequest(
            request_id=0, session_id=0, func_name=func_name, func_code=None
        )
        node = cls(pseudo_native_func)
        node.input_vars = input_vars
        node.output_vars = output_vars
        return node

    # ---------- Polling ----------

    async def wait_ready(self) -> None:
        """Wait until all inputs are ready."""

        for sv in self.input_vars.values():
            await sv.wait_ready()

    async def wait_activated(self) -> None:
        """Wait until the node is activated."""

        coros = [sv.wait_activated() for sv in self.output_vars.values()]
        await asyncio.wait(coros, return_when=asyncio.FIRST_COMPLETED)


# Types

SVProducer = Union[PlaceholderGen, NativeFuncNode]
