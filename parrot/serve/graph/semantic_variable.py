# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional
from asyncio import Event

from parrot.exceptions import parrot_assert
from .perf_criteria import PerformanceCriteria


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
        var_id: str,
        is_constant_prefix: bool,
        seed: int,
    ) -> None:
        # Basic Info
        self.name = name
        self.id = var_id
        self.seed = seed  # A seed for generating the var_id. For id recycling.
        self.is_constant_prefix = (
            is_constant_prefix  # Whether this SV is a constant prefix.
        )

        # Text content.
        self._content: Optional[str] = None

        # Events
        self._ready_event: Event = Event()  # Ready event means the content is ready.

        # Producer of this SV. It must be a PlaceholderGen node or a NativeFuncNode.
        self._producer: Optional["SVProducer"] = None

        # Consumers of this SV. It must be Fill nodes.
        self._consumers: List["PlaceholderFill"] = []

        # Activate
        self._activated_event: Event = Event()
        # Performance criteria of "get" to this SV.
        self._criteria: Optional[PerformanceCriteria] = None
        # Distance to "get" node.
        self._depth: int = 99999

    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def set(self, content: str) -> None:
        """Set the content of the semantic variable."""

        assert self._content is None, f"This semantic variable (id={self.id}) is filled"
        self._content = content
        self._ready_event.set()

    def get(self) -> str:
        """Get the content of the semantic variable."""

        parrot_assert(
            self.is_ready(), f"This semantic variable (id={self.id}) is not ready"
        )

        return self._content

    async def wait_ready(self) -> None:
        """Wait until the content of this SV is ready."""

        await self._ready_event.wait()

    def assign_producer(self, producer: "SVProducer") -> None:
        """Assign the producer of this SV. This will add some edges in the graph."""

        parrot_assert(self._producer is None, "This SV already has a producer")
        self._producer = producer

    def add_consumer(self, consumer: "PlaceholderFill") -> None:
        """Add a consumer of this SV. This will add some edges in the graph."""

        self._consumers.append(consumer)

    @property
    def has_producer(self) -> bool:
        return self._producer is not None

    def get_producer(self) -> Optional["SVProducer"]:
        return self._producer

    def get_consumers(self) -> List["PlaceholderFill"]:
        return self._consumers

    def activate(self, criteria: PerformanceCriteria, depth: int) -> None:
        """Activate the Semantic Variable with a given PerformanceCriteria."""

        parrot_assert(
            not self.is_activated,
            "Semantic Variable has been activated.",
        )
        self._criteria = criteria
        self._depth = depth
        self._activated_event.set()

    @property
    def is_activated(self) -> bool:
        return self._activated_event.is_set()

    async def wait_activated(self) -> None:
        await self._activated_event.wait()

    @property
    def criteria(self) -> PerformanceCriteria:
        parrot_assert(self.is_activated, "Semantic Variable has not been activated.")
        return self._criteria

    @property
    def depth(self) -> int:
        parrot_assert(self.is_activated, "Semantic Variable has not been activated.")
        return self._depth
