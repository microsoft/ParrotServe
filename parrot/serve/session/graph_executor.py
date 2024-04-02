# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict

from parrot.utils import get_logger

from parrot.serve.graph import (
    ComputeGraph,
    RequestChain,
    CompletionChain,
)

from ..scheduler.global_scheduler import GlobalScheduler
from ..engine_manager import EngineManager
from ..tokenizer_wrapper import TokenizersWrapper


logger = get_logger("GraphExecutor")


class GraphExecutor:
    """
    GraphExecutor in a session polls CompletionChain to GlobalScheduler,
    waiting for scheduling and execute it.
    """

    def __init__(
        self,
        session_id: int,
        scheduler: GlobalScheduler,
        engine_mgr: EngineManager,
        tokenizers_wrapper: TokenizersWrapper,
    ):
        # ---------- Basic Info ----------
        self.session_id = session_id
        self.graph = ComputeGraph()

        # ---------- Global Components ----------
        self.scheduler = scheduler
        self.engine_mgr = engine_mgr
        self.tokenizers_wrapper = tokenizers_wrapper

    def add_request(self, request_chain: RequestChain) -> None:
        """Add a request to the graph and assign a coroutine to the request."""

        # Insert the request chain into the graph.

    async def _execute_coroutine(self, completion_chain: CompletionChain) -> None:
        """Coroutine for executing a CompletionChain."""

        for node in completion_chain.iter_fill():
            await node.wait_ready()  # Blocking
