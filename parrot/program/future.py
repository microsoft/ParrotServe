# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Optional


class Future:
    """Represents a string which will be filled in the Future.

    It's like "Future" in the Python asynchronous programming, or "Promise" in JavaScript.
    As its name suggests, it's a placeholder for the content to be filled in the future.

    It also corresponds to a Input/Output semantic variable in Parrot System.
    """

    _counter = 0
    _virtual_machine_env: Optional["VirtualMachine"] = None

    def __init__(
        self,
        name: Optional[str] = None,
        content: Optional[str] = None,
    ):
        self.id = self._increment()
        self.name = name if name is not None else f"v{self.id}"
        self.content = content

    @classmethod
    def _increment(cls) -> int:
        cls._counter += 1
        return cls._counter

    def __repr__(self) -> str:
        if self.ready:
            return f"Future(name={self.name}, id={self.id}, content={self.content})"
        return f"Future(name={self.name}, id={self.id})"

    # ---------- Public Methods ----------

    @property
    def ready(self) -> bool:
        return self.content is not None

    def get(self) -> str:
        """Public API: (Blocking) Get the content of the future."""

        if self.ready:
            return self.content
        content = self._virtual_machine_env._placeholder_fetch(self.id)
        return content

    async def aget(self) -> str:
        """Public API: (Asynchronous) Get the content of the future."""

        if self.ready:
            return self.content
        content = await self._virtual_machine_env._aplaceholder_fetch(self.id)
        return content
