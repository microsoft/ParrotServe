# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional, Dict

from parrot.sampling_config import SamplingConfig
from parrot.exceptions import parrot_assert

from .request import RequestPlaceholder
from .semantic_variable import SemanticVariable


class BaseNode:
    """Represent a computational node in the graph."""

    def __init__(self):
        self.sv: Optional[SemanticVariable] = None
        self.id_in_graph: Optional[int] = None
        self.completion_chain: Optional["CompletionChain"] = None
        self.request_chain: Optional["RequestChain"] = None

        # Edge type A: Fill -> Fill -> Fill -> Gen -> Fill -> Fill -> Gen -> ...
        self.edge_a_prev_node: Optional["BaseNode"] = None
        self.edge_a_next_node: Optional["BaseNode"] = None

    @property
    def has_var(self) -> bool:
        """Whether the node has a SV."""

        return self.sv is not None

    @property
    def is_inserted(self) -> bool:
        """Whether the node is inserted into the graph."""

        return self.id_in_graph is not None

    @property
    def is_gen(self) -> bool:
        """Whether the node is a Gen node."""

        return isinstance(self, PlaceholderGen)

    @property
    def has_placeholder(self) -> bool:
        """Whether the node has a placeholder."""

        return not isinstance(self, ConstantFill)

    @property
    def edge_b_prev_node(self) -> Optional["BaseNode"]:
        """Edge type B: prev node. 0 or 1."""

        parrot_assert(self.is_inserted, "Should be inserted before get DAG info.")
        return self.sv.producer

    @property
    def edge_b_next_nodes(self) -> List["BaseNode"]:
        """Edge type B: next node. Only Gen node has multiple next nodes."""

        parrot_assert(self.is_inserted, "Should be inserted before get DAG info.")
        if not isinstance(self, PlaceholderGen):
            return []
        return self.sv.consumers

    async def wait_ready(self):
        """Wait until the node is ready. A node is ready if all its inputs are ready.

        To be specific, a node in our graph can only have at most 2 inputs:
        - Predecessor in edge type A (previous Fill)
        - Predcessor in edge type B (Gen in the same SV)

        The node is ready iff. all its predecessors' SVs are ready.
        """

        if self.edge_a_prev_node is not None:
            await self.edge_a_prev_node.sv.wait_ready()

        if self.edge_b_prev_node is not None:
            await self.edge_b_prev_node.sv.wait_ready()

    @property
    def sv_name(self) -> str:
        # parrot_assert(self.has_var, "This node has no SV.")
        if not self.has_var:
            return "(no SV)"
        return self.sv.name

    @property
    def sv_id(self) -> str:
        # parrot_assert(self.has_var, "This node has no SV.")
        if not self.has_var:
            return "(no SV)"
        return self.sv.sv_id

    def get(self) -> str:
        """Get the content of the node."""

        return self.sv.get()

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
            return f"{self.id_in_graph}: " + repr
        return repr

    def short_repr(self) -> str:
        """Short representation of the node."""

        return self._short_repr_add_graph_id("BaseNode")


class ConstantFill(BaseNode):
    """Represent a fill node (constant) in the graph."""

    def __init__(self, constant_text: str):
        super().__init__()
        self.constant_text = constant_text

    def _get_display_elements(self) -> Dict:
        return {
            "sv_name": self.sv_name,
            "sv_id": self.sv_id,
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


class PlaceholderFill(BaseNode):
    """Represent a fill node (placeholder) in the graph."""

    def __init__(self, placeholder: RequestPlaceholder):
        super().__init__()
        self.placeholder = placeholder

    def _get_display_elements(self) -> Dict:
        return {
            "sv_name": self.sv_name,
            "sv_id": self.sv_id,
            "placeholder_name": self.placeholder.name,
        }

    def short_repr(self) -> str:
        return self._short_repr_add_graph_id(
            f"PlaceholderFill({self.placeholder.name})"
        )


class PlaceholderGen(BaseNode):
    """Represent a gen node (placeholder, actually it must be) in the graph."""

    def __init__(self, placeholder: RequestPlaceholder):
        super().__init__()
        self.placeholder = placeholder

    @property
    def sampling_config(self) -> SamplingConfig:
        return self.placeholder.sampling_config

    def _get_display_elements(self) -> Dict:
        return {
            "sv_name": self.sv_name,
            "sv_id": self.sv_id,
            "placeholder_name": self.placeholder.name,
            "sampling_config": self.sampling_config,
        }

    def short_repr(self) -> str:
        return self._short_repr_add_graph_id(f"PlaceholderGen({self.placeholder.name})")

    # async def wait_ready(self):
    #     """NOTE(chaofan): We don't need to wait Gen to be ready."""
    #     pass
