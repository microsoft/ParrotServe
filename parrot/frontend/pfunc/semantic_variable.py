# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional

from parrot.utils import get_logger

from .perf_criteria import PerformanceCriteria


logger = get_logger("PFunc Variable")


class SemanticVariable:
    """Maintain an object in PFunc frontend, which represents a semantic variable in ServeLayer.

    The register/unregister of the variable is managed collaboratively by VM and Python Interpreter.
    """

    _var_counter = 0
    _virtual_machine_env: Optional["VirtualMachine"] = None

    def __init__(
        self,
        name: Optional[str] = None,
        content: Optional[str] = None,
    ) -> None:
        if name is None:
            self.name = f"var_{SemanticVariable._var_counter}"
            SemanticVariable._var_counter += 1
        else:
            self.name = name

        if SemanticVariable._virtual_machine_env is not None:
            self.id = self._register_semantic_variable(self.name)

        self.content = content
        if self.content is not None:
            self._set_semantic_variable(self.content)

    def __repr__(self) -> str:
        if self.ready:
            return f"SemanticVariable(name={self.name}, id={self.id}, content={self.content})"
        return f"SemanticVariable(name={self.name}, id={self.id})"

    # ---------- VM Env Methods ----------

    def _has_vm_env(self) -> bool:
        return SemanticVariable._virtual_machine_env is not None

    def _register_semantic_variable(self, name: str) -> str:
        if self._has_vm_env():
            return self._virtual_machine_env.register_semantic_variable_handler(
                self.name
            )
        else:
            logger.warning(
                f"VM environment is not set. Not register variable (name={name})."
            )
            return str(self._var_counter)

    def _set_semantic_variable(self, content: str) -> None:
        if self._has_vm_env():
            self._virtual_machine_env.set_semantic_variable_handler(self.id, content)
        else:
            logger.warning(
                f"VM environment is not set. Set variable (id={self.id}) failed."
            )

    def _get_semantic_variable(self, criteria: PerformanceCriteria) -> str:
        if self._has_vm_env():
            return self._virtual_machine_env.get_semantic_variable_handler(
                self.id, criteria
            )
        else:
            logger.warning(
                f"VM environment is not set. Get variable (id={self.id}) failed."
            )
            return ""

    async def _aget_semantic_variable(self, criteria: PerformanceCriteria) -> str:
        if self._has_vm_env():
            return await self._virtual_machine_env.aget_semantic_variable_handler(
                self.id, criteria
            )
        else:
            logger.warning(
                f"VM environment is not set. Get variable (id={self.id}) failed."
            )
            return ""

    # ---------- Public Methods ----------

    @property
    def ready(self) -> bool:
        return self.content is not None

    def set(self, content: str) -> None:
        """Set the content of variable."""

        assert (not self.ready, "The variable can't be set repeatedly.")

        self._set_semantic_variable(self.id, content)
        self.content = content
        return

    def get(self, criteria: PerformanceCriteria) -> str:
        """(Blocking) Get the content of the variable."""

        if self.ready:
            return self.content

        self.content = self._get_semantic_variable(criteria)
        return self.content

    async def aget(self, criteria: PerformanceCriteria) -> str:
        """(Asynchronous) Get the content of the variable."""

        if self.ready:
            return self.content

        content = await self._aget_semantic_variable(criteria)
        return content
