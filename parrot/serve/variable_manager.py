# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import uuid
import time

from typing import List, Optional, Dict

from parrot.utils import RecyclePool
from parrot.exceptions import parrot_assert, ParrotCoreUserError

from parrot.serve.graph import (
    SemanticVariable,
    RequestChain,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
)


class SemanticVariableNamespace:
    """A namespace of Semantic Variables, giving a unique id to each SV.

    The guideline is to hash the most important information of the SV to a single ID.
    There are majorly two types of SVs:
    - Constants: The content is the most important information. We use content-based hashing.
    - Placeholders: The placeholder itself is the most important information. We allocate a seed for
        each placeholder.
    """

    def __init__(self) -> None:
        # Variables: sv_id -> variable
        self.vars: Dict[str, SemanticVariable] = {}

        # Name Generating
        # Seed is for generating unique names.
        self.seed_pool = RecyclePool("SemanticVariable")
        self.namespace_uuid = uuid.uuid4()  # A UUID object.

    def _get_hashed_sv_id(self, content: str) -> str:
        return str(
            uuid.uuid3(
                namespace=self.namespace_uuid,
                name=str(content),
            )
        )

    def get_var_by_content(self, content: str) -> Optional[SemanticVariable]:
        """Get a Semantic Variable by content."""

        sv_id = self._get_hashed_sv_id(content)

        return self.vars.get(sv_id)

    def new_var_by_content(self, content: str, is_global: bool) -> SemanticVariable:
        """Create a new Semantic Variable."""

        seed = -1
        hash_name = content

        sv_id = self._get_hashed_sv_id(hash_name)

        parrot_assert(sv_id not in self.vars, "SV ID already exists.")

        sv = SemanticVariable(
            name="constant", sv_id=sv_id, is_global=is_global, seed=seed
        )
        self.vars[sv_id] = sv

        return sv

    def new_var_by_name(self, name: str, is_global: bool) -> SemanticVariable:
        """Create a new Semantic Variable."""

        seed = self.seed_pool.allocate()
        hash_name = str(seed)

        sv_id = self._get_hashed_sv_id(hash_name)

        parrot_assert(sv_id not in self.vars, "SV ID already exists.")

        sv = SemanticVariable(name=name, sv_id=sv_id, is_global=is_global, seed=seed)
        self.vars[sv_id] = sv

        return sv

    def free_var(self, sv: SemanticVariable) -> None:
        """Free a Semantic Variable."""

        parrot_assert(sv.sv_id in self.vars, "SV ID does not exist.")

        self.seed_pool.free(sv.seed)
        self.vars.pop(sv.sv_id)


class SemanticVariableManager:
    """Manager for all Semantic Variables used in the system."""

    def __init__(self) -> None:
        # ---------- Namespace ----------
        self.global_namespace = SemanticVariableNamespace()

        # session_id -> namespace
        self.local_namespace: Dict[int, SemanticVariableNamespace] = {}

        # ---------- Global Var Management ----------

        # sv_id -> last_access_time
        self.gvar_last_access_time: Dict[str, float] = {}

    # ---------- Internal methods ----------

    def _get_global_var(self, content: str) -> SemanticVariable:
        """Get a global variable."""

        gvar = self.global_namespace.get_var_by_content(content)
        if gvar is None:
            gvar = self.global_namespace.new_var_by_content(content, is_global=True)
        self.gvar_last_access_time[gvar.sv_id] = time.perf_counter_ns()
        return gvar

    def _create_local_var_by_content(
        self, session_id: int, content: str
    ) -> SemanticVariable:
        namespace = self.local_namespace[session_id]
        lvar = namespace.new_var_by_content(content, is_global=False)
        return lvar

    def _create_local_var_by_name(self, session_id: int, name: str) -> SemanticVariable:
        namespace = self.local_namespace[session_id]
        lvar = namespace.new_var_by_name(name, is_global=False)
        return lvar

    def _get_local_var_by_id(self, session_id: int, sv_id: str) -> SemanticVariable:
        namespace = self.local_namespace[session_id]
        lvar = namespace.vars.get(sv_id)
        parrot_assert(lvar is not None, "Local variable does not exist.")
        return lvar

    # ---------- Public methods ----------

    def register_local_var_space(self, session_id: int) -> None:
        """Register a local namespace."""

        parrot_assert(
            session_id not in self.local_namespace,
            "Session ID already exists.",
        )

        self.local_namespace[session_id] = SemanticVariableNamespace()

    def free_local_var_space(self, session_id: int) -> None:
        """Free a local namespace."""

        parrot_assert(
            session_id in self.local_namespace,
            "Session ID does not exist.",
        )

        self.local_namespace.pop(session_id)

    def get_var(self, session_id: int, sv_id: str) -> SemanticVariable:
        """Get a Semantic Variable by ID.

        Args:
            session_id: int. The session ID.
            sv_id: str. The Semantic Variable ID.
        """

        if sv_id in self.global_namespace.vars:
            return self.global_namespace.vars[sv_id]

        parrot_assert(
            session_id in self.local_namespace,
            f"Local namespace of {session_id} does not exist.",
        )

        if sv_id in self.local_namespace[session_id].vars:
            return self.local_namespace[session_id].vars[sv_id]

        raise ParrotCoreUserError(ValueError(f"Unknown Semantic Variable ID: {sv_id}"))

    def create_vars_for_request(
        self, session_id: int, request_chain: RequestChain
    ) -> None:
        """Create all the Semantic Variables in the request chain.

        Args:
            session_id: int. The session ID.
            request_chain: RequestChain. The request chain.
        """

        constant_prefix_flag: bool = True
        for node in request_chain.iter():
            if node.has_placeholder:
                constant_prefix_flag = False

            if isinstance(node, ConstantFill):
                if constant_prefix_flag:
                    gvar = self._get_global_var(content=node.constant_text)
                    node.sv = gvar
                else:
                    lvar = self._create_local_var_by_content(
                        session_id=session_id,
                        content=node.constant_text,
                    )
                    node.sv = lvar
            elif isinstance(node, PlaceholderFill):
                lvar = self._get_local_var_by_id(
                    session_id=session_id,
                    sv_id=node.placeholder.var_id,
                )
                node.sv = lvar
            elif isinstance(node, PlaceholderGen):
                lvar = self._create_local_var_by_name(
                    session_id=session_id,
                    name=node.placeholder.name,
                )
                node.sv = lvar
            else:
                parrot_assert(
                    False,
                    "Unknown node type.",
                )

        request_chain.sv_created = True
