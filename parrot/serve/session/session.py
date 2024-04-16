# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum
import asyncio
from typing import List, Dict, Optional
from queue import Queue

from parrot.utils import get_logger
from parrot.exceptions import ParrotCoreUserError, parrot_assert

from parrot.serve.graph import (
    ChunkedSemanticCallRequest,
    RequestChain,
)

from parrot.serve.backend_repr import Context
from parrot.serve.scheduler import TaskCreator, GlobalScheduler

from ..prefix_matcher import PrefixMatcher
from ..variable_manager import SemanticVariableManager
from ..engine_manager import EngineManager
from ..tokenizer_wrapper import TokenizersWrapper
from ..context_manager import ServeCoreContextManager
from .graph_executor import GraphExecutor


logger = get_logger("Session")


class SessionStatus(Enum):
    RUNNING = 0  # The session is running
    DEAD = 1  # The session is dead, i.e. the program (in the frontend) is disconnected / timed out
    BAD = 2  # The session is bad, i.e. the session throws an exception during execution


class Session:
    """
    A session is an abstraction of a program interacting with the OS: When a program connected to the OS,
    a session will be created for it. The session will be removed when the program is disconnected/timed out.

    A session has its own ComputeGraph and GraphExecutor.
    """

    def __init__(
        self,
        session_id: int,
        life_span: int,
        prefix_matcher: PrefixMatcher,
        task_creator: TaskCreator,
        scheduler: GlobalScheduler,
        var_mgr: SemanticVariableManager,
        engine_mgr: EngineManager,
        context_mgr: ServeCoreContextManager,
        tokenizers_wrapper: TokenizersWrapper,
    ):
        # ---------- Basic Info ----------
        self.session_id = session_id
        self.life_span = life_span  # In seconds

        # ---------- Global Components ----------
        self._prefix_matcher = prefix_matcher
        self._var_mgr = var_mgr
        self._context_mgr = context_mgr

        # ---------- Executor ----------
        self.executor = GraphExecutor(
            session_id=session_id,
            task_creator=task_creator,
            scheduler=scheduler,
            engine_mgr=engine_mgr,
            context_mgr=context_mgr,
            tokenizers_wrapper=tokenizers_wrapper,
        )

        # ---------- Runtime Status ----------
        self.status = SessionStatus.RUNNING

        # ---------- Requests ----------
        self._request_id_counter = (
            0  # We don't use RecyclePool since the lifetime of a session is short.
        )

        self._register_session_resources()

    # ---------- Internal methods ----------

    # ---------- Status Methods ----------

    @property
    def is_running(self) -> bool:
        return self.status == SessionStatus.RUNNING

    # ---------- Interfaces to ServeCore ----------

    def add_request(self, request_payload: Dict) -> Dict:
        """Add a request to the session and assign a coroutine to the request.

        Args:
            request_payload (Dict): The request payload.

        Returns:
            int: The request id.
            List: The placeholder mapping.
        """

        # Get the request id.
        request_id = self._request_id_counter
        self._request_id_counter += 1

        # Convert the request to a ChunkedRequest.
        chunked_request = ChunkedSemanticCallRequest.parse_from_payload(
            request_id=request_id,
            session_id=self.session_id,
            payload=request_payload,
        )

        # Prefix matching and splitting.

        # Convert the ChunkedRequest to a RequestChain.
        request_chain = RequestChain.from_chunked_request(chunked_request)

        # Assign Semantic Variables to the RequestChain.
        self._var_mgr.create_vars_for_request(
            session_id=self.session_id, request_chain=request_chain
        )

        # Add the request to the executor.
        self.executor.add_request(request_chain=request_chain)

        # It must be inserted. So we can get the mapping.
        placeholders_mapping = request_chain.get_placeholders_mapping()

        return {
            "request_id": request_id,
            "placeholders_mapping": placeholders_mapping,
        }

    # def execute_native_call(self, call: NativeCall):
    #     async def _execute_body(func, *args):
    #         return func(*args)

    #     async def _execute_main():
    #         try:
    #             # Mark all placeholders as start
    #             for _, value in call.bindings.items():
    #                 if isinstance(value, SVPlaceholder):
    #                     value.start_event.set()

    #             # Wait all inputs to be ready
    #             args = []
    #             for name, value in call.bindings.items():
    #                 if call.func.params_map[name].is_output:
    #                     continue
    #                 elif isinstance(value, SVPlaceholder):
    #                     args.append(await value.get())  # Maybe block here
    #                     continue
    #                 else:
    #                     args.append(value)

    #             # Execute the native call
    #             native_pyfunc = call.func.get_pyfunc()
    #             result = await asyncio.wait_for(
    #                 _execute_body(native_pyfunc, *args),
    #                 call.func.metadata.timeout,
    #             )

    #             # Set the output
    #             call.output_vars[0].set(result)
    #         except BaseException as e:
    #             self.exception_interrupt(e)

    #     create_task_in_loop(_execute_main(), fail_fast=False)

    def _register_session_resources(self) -> None:
        self._context_mgr.register_session_contexts(session_id=self.session_id)
        self._var_mgr.register_local_var_space(session_id=self.session_id)

    def free_session_resources(self) -> None:
        """Free the session and all its resources."""

        # Free the contexts of session.
        self._context_mgr.free_session_contexts(session_id=self.session_id)

        # Free the local var space of the session.
        self._var_mgr.free_local_var_space(session_id=self.session_id)

        logger.info(
            f"Free Session(session_id={self.session_id}) with running tasks num: {0}"
        )
