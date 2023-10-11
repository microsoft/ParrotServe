from typing import Optional, Coroutine
import asyncio
from asyncio import Event


class Future:
    """Represents a string which will be filled in the Future.

    It's like "Future" in the Python asynchronous programming, or "Promise" in JavaScript.
    As its name suggests, it's a placeholder for the content to be filled in the future.

    When as inputs, the data source can be a str or a Coroutine.
    - If it's a str, the future is filled immediately.
    - If it's an Coroutine, the future is filled when the Coroutine is done. (The Coroutine is
      usually an external async function call from other asynchronous libraries.)

    When as the middle results, these two sources are all set as None. And the data comes from
    the previous function call.
    """

    _counter = 0

    def __init__(
        self,
        content: Optional[str] = None,
        coroutine: Optional[Coroutine] = None,
    ):
        assert (
            content is None or coroutine is None
        ), "Cannot set both content and Coroutine"

        self.id = self._increment()
        self.content = content
        self.coroutine = coroutine
        self.ready_event: Event = Event()
        if self.content:
            self.ready_event.set()

        self.assign_callbacks = []

    @classmethod
    def _increment(cls) -> int:
        cls._counter += 1
        return cls._counter

    def __repr__(self) -> str:
        if self.ready:
            return f"Future(id={self.id}, content={self.content})"
        return f"Future(id={self.id})"

    def _set(self, content: str, no_callback=False):
        """Internal: set the content of the future with callback-invoking."""
        assert self.content is None, "This future is filled"
        self.content = content
        self.ready_event.set()

        if not no_callback:
            for callback in self.assign_callbacks:
                callback()

    async def _wait_content(self):
        assert (
            self.coroutine is not None
        ), "This future doesn't has a Coroutine data source"
        self._set(await self.coroutine)

    # Public APIs

    @property
    def is_input(self) -> bool:
        return self.content is not None or self.coroutine is not None

    @property
    def is_middle_node(self) -> bool:
        return self.content is None and self.coroutine is None

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def set(self, content: str):
        """Public API: Set the content of the future. We don't use it usually."""

        self._set(content)

    async def get(self) -> str:
        """Public API: (Asynchronous) Get the content of the future."""

        if self.coroutine is not None:
            await self._wait_content()

        await self.ready_event.wait()
        return self.content
