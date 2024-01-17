# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Optional, Dict

from parrot.protocol.sampling_config import SamplingConfig
from parrot.exceptions import parrot_assert

from .request import RequestPlaceholder
from .semantic_variable import SemanticVariable, SemanticVariableNamespace


# ---------- Graph ----------


class BaseNode:
    """Represent a computational node in the graph."""

    def __init__(self):
        # Basic Info
        self.sv: Optional[SemanticVariable] = None
        self.gen_task: Optional["GenTask"] = None

        # Graph

        # Edge type A: Fill -> Fill -> Fill -> Gen -> Fill -> Fill -> Gen -> ...
        self.edge_a_prev_node: Optional["BaseNode"] = None
        self.edge_a_next_node: Optional["BaseNode"] = None

        # Flags
        self.is_scheduled = (
            False  # Whether the GenTask of this node is scheduled to an engine.
        )
        self.is_finished = False  # Whether the GenTask of this node is finished.

    @property
    def is_inserted(self) -> bool:
        """Whether the node is inserted into the graph."""

        return self.sv is not None

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

    @property
    def name(self) -> str:
        return self.sv.name

    async def get(self) -> str:
        """Get the content of the node."""

        return await self.sv.get()


class ConstantFill(BaseNode):
    """Represent a fill node (constant) in the graph."""

    def __init__(self, constant_text: str):
        self.constant_text = constant_text


class PlaceholderFill(BaseNode):
    """Represent a fill node (placeholder) in the graph."""


class PlaceholderGen(BaseNode):
    """Represent a gen node (placeholder, actually it must be) in the graph."""

    def __init__(
        self,
        sampling_config: SamplingConfig,
    ):
        self.sampling_config = sampling_config


class StaticGraph:
    """Computational graph of LLM requests linked by Semantic Variables."""

    def __init__(self):
        self.nodes: List[BaseNode] = []

        # ---------- Semantic Vars ----------
        self.vars: Dict[str, SemanticVariable] = {}  # id -> sv
        self.sv_namespace = SemanticVariableNamespace()

    def _get_new_var(self, sv_name: str, producer: Optional[BaseNode] = None):
        """Get a new Semantic Variable."""

        sv_id = self.sv_namespace.get_new_id()
        sv = SemanticVariable(name=sv_name, sv_id=sv_id, producer=producer)
        self.vars[sv_id] = sv
        return sv

    def insert_node(
        self, node: BaseNode, placeholder: Optional[RequestPlaceholder] = None
    ):
        """Insert a node into the graph. The node must be created
        without a SV and graph informaton.
        """

        self.nodes.append(node)

        if isinstance(node, ConstantFill):
            # Constant node has no topology in the graph.
            sv = self._get_new_var(sv_name="constant")
            sv.set(node.constant_text)
            sv.consumers.append(node)
            node.sv = sv
        elif isinstance(node, PlaceholderFill):
            parrot_assert(placeholder is not None, "Placeholder arg is None.")

            # For PlaceholderFill, the SV may be reused if it is already created.
            if placeholder.const_value is not None:
                sv = self._get_new_var(sv_name=placeholder.name)
                sv.set(placeholder.const_value)
            elif placeholder.var_id is not None:
                sv = self.vars[placeholder.var_id]
            else:
                sv = self._get_new_var(sv_name=placeholder.name)
            sv.consumers.append(node)
            node.sv = sv

            # Link with producer of this SV.
        elif isinstance(node, PlaceholderGen):
            # For PlaceholderGen, the SV must be created.
            # And we don't link with other SVs, because it will be linked by the subsequent Fill nodes.
            parrot_assert(placeholder is not None, "Placeholder arg is None.")
            sv = self._get_new_var(sv_name=placeholder.name, producer=node)
            node.sv = sv
