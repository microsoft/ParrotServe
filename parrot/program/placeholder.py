from typing import Optional
from asyncio import Event


class Placeholder:
    """Placeholder for context variables.

    It's like "Promise[Content]" in the asynchronous programming.
    """

    def __init__(self, name: str, content: Optional[str]):
        self.name = name
        self.content = content
        self.ready_event: Event = Event()
        if self.content:
            self.ready_event.set()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def assign(self, content: str):
        assert self.content is None, "This placeholder is filled"
        self.content = content
        self.ready_event.set()

    def get(self):
        while not self.ready:
            """Blocking"""

        return self.content

    async def aget(self):
        await self.ready_event.wait()

        assert self.ready
        return self.content
