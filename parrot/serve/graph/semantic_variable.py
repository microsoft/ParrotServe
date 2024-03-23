# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional
from asyncio import Event

from parrot.exceptions import parrot_assert


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

    For input/output SVs, they are like "Future" in the Python asynchronous programming,
    or "Promise" in JavaScript.

    The motivation is that the prompt itself is structural, and can be split into
    different independent parts with different semantic purposes.
    """

    def __init__(
        self,
        name: str,
        sv_id: str,
        is_global: bool,
        seed: int,
    ) -> None:
        # Basic Info
        self.name = name
        self.sv_id = sv_id
        self.seed = seed  # A seed for generating the sv_id. For id recycling.

        # Text content.
        self.content: Optional[str] = None

        # Events
        self.ready_event: Event = Event()

        # Producer of this SV. It must be a PlaceholderGen node.
        self.producer: Optional["PlaceholderGen"] = None

        # Consumers of this SV. It must be Fill nodes.
        self.consumers: List["PlaceholderFill"] = []

        # Whether this SV is global.
        # If it is global, it will be shared across different sessions.
        self.is_global = is_global

    @property
    def ready(self) -> bool:
        return self.ready_event.is_set()

    def set(self, content: str):
        """Set the content of the placeholder."""

        assert self.content is None, "This placeholder is filled"
        self.content = content
        self.ready_event.set()

    def get(self) -> str:
        """Get the content of the placeholder."""

        parrot_assert(self.ready, "This placeholder is not ready")

        return self.content

    async def wait_ready(self):
        """Wait until the content of this SV is ready."""

        await self.ready_event.wait()

    def assign_producer(self, producer: "PlaceholderGen"):
        """Assign the producer of this SV. This will add some edges in the graph."""

        parrot_assert(self.producer is None, "This SV already has a producer")
        self.producer = producer

    def remove_producer(self):
        """Remove the producer of this SV because the content is already generated.
        This will remove some edges in the graph.
        """

        parrot_assert(self.ready, "This SV is not ready")
        self.producer = None

    def add_consumer(self, consumer: "PlaceholderFill"):
        """Add a consumer of this SV. This will add some edges in the graph."""

        self.consumers.append(consumer)

    def remove_consumer(self, consumer: "PlaceholderFill"):
        """Remove a consumer of this SV because the content is already consumed.
        This will remove some edges in the graph.
        """

        parrot_assert(self.ready, "This SV is not ready")
        self.consumers.remove(consumer)
