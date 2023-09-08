from typing import Optional, List
from asyncio import Event


class Placeholder:
    """Placeholder for context variables.

    It's like "Future" in the Python asynchronous programming.
    """

    _counter = 0

    def __init__(self, name: Optional[str], content: Optional[str]):
        self.name = name if name is not None else self._get_default_name()
        self.content = content
        self.ready_event: Event = Event()
        if self.content:
            self.ready_event.set()

        self.assign_callbacks = []

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def assign(self, content: str):
        assert self.content is None, "This placeholder is filled"
        self.content = content

        for callback in self.assign_callbacks:
            callback()

        self.ready_event.set()

    async def get(self):
        await self.ready_event.wait()
        return self.content

    @classmethod
    def _get_default_name(cls) -> str:
        cls._counter += 1
        return f"placeholder_{cls._counter}"
