# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import uuid

from typing import List, Optional
from asyncio import Event

from parrot.exceptions import parrot_assert


# ---------- Placeholder ----------


class TextPlaceholder:
    """TextPlaceholder holds data and maintains signals for a placeholder.

    It's like "Future" in the Python asynchronous programming, or "Promise" in JavaScript.
    As its name suggests, it's a placeholder for the content to be filled in the future.

    It also corresponds to a Input/Output semantic variable in Parrot System.
    """

    def __init__(self, sv: "SemanticVariable"):
        self.sv = sv
        self.content: Optional[str] = None
        self.tokens_id_holders: List["TokensIdHolder"] = []

        # Events
        self.ready_event: Event = Event()

    @property
    def id(self) -> str:
        return self.sv.sv_id

    @property
    def name(self) -> str:
        return self.sv.name

    def __repr__(self) -> str:
        if self.ready:
            return (
                f"Placeholder(name={self.name}, id={self.id}, content={self.content})"
            )
        return f"Placeholder(name={self.name}, id={self.id})"

    def set(self, content: str):
        """Set the content of the placeholder."""

        assert self.content is None, "This placeholder is filled"
        self.content = content
        self.ready_event.set()

        # Sync results to token holders
        for tokens_id_holder in self.tokens_id_holders:
            tokens_id_holder.sync_from_placeholder()

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    async def get(self) -> str:
        """Get the content of the placeholder."""

        await self.ready_event.wait()
        return self.content


# ---------- SemanticVariable ----------


class SemanticVariable:
    """Semantic Variable: the core abstraction in Parrot system.

    Its main purpose is to chunk a LLM request into smaller pieces, so that
    we can do fine-grained management and optimization.

    Definition: a Semantic Variable (SV) is a part of prompts with specific semantic
    purpose. A SV can be:
    - 1. A system prompt of a request (Also called prefix).
    - 2. A user-input of a request (Also called an input / a parameter of a function).
    - 3. An output location of a request (Also called a return value of a function).
    - 4. A communication port of two LLM Agents.
    - 5. A few-shot example of a request.
    - ...

    The motivation is that the prompt itself is structural, and can be split into
    different independent parts with different semantic purposes.
    """

    def __init__(self, name: str, sv_id: str, producer: Optional["BaseNode"] = None):
        # Basic Info
        self.name = name
        self.sv_id = sv_id
        self.text_placeholder = TextPlaceholder(sv=self)

        # Producer of this SV. It must be a PlaceholderGen node.
        self.producer = producer

        # Consumers of this SV. It must be Fill nodes.
        self.consumers: List["PlaceholderFill"] = []

    def set(self, content: str):
        """Set the content of this SV."""

        self.text_placeholder.set(content)

    async def get(self):
        """Get the content of this SV."""

        return await self.text_placeholder.get()

    @property
    def ready(self) -> bool:
        return self.text_placeholder.ready

    def produce_finish(self):
        """Remove the producer of this SV because the content is already generated.
        This will remove some edges in the graph.
        """

        parrot_assert(self.ready, "This SV is not ready")
        self.producer = None


class SemanticVariableNamespace:
    """A namespace of Semantic Variables, giving a unique id to each SV."""

    def __init__(self):
        self._counter = 0  # This is not recycled. It just keeps increasing.
        self._namespace_uuid = uuid.uuid4()

    def get_new_id(self) -> str:
        """Get a new id for a SV."""

        self._counter += 1
        # return str(
        #     uuid.uuid3(
        #         namespace=self._namespace_uuid,
        #         name=str(self._counter),
        #     )
        # )
        return str(self._counter)  # For easier debugging
