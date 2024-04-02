# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum
import asyncio
from typing import List, Dict, Optional
from queue import Queue

from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError, parrot_assert

from parrot.serve.graph import (
    ChunkedRequest,
    RequestChain,
)

from parrot.serve.backend_repr import Context

from ..prefix_matcher import PrefixMatcher
from ..scheduler.global_scheduler import GlobalScheduler
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
        self.prefix_matcher = prefix_matcher
        self.scheduler = scheduler
        self.var_mgr = var_mgr
        self.engine_mgr = engine_mgr
        self.context_mgr = context_mgr
        self.tokenizers_wrapper = tokenizers_wrapper

        # ---------- Executor ----------
        self.executor = GraphExecutor(
            session_id=session_id,
            scheduler=scheduler,
            engine_mgr=engine_mgr,
            tokenizers_wrapper=tokenizers_wrapper,
        )

        # ---------- Runtime Status ----------
        self.status = SessionStatus.RUNNING
        self.bad_exception: Optional[Exception] = None

        # Register local spaces
        # self.context_mgr.
        self.var_mgr.register_local_var_space(session_id=self.session_id)

    # ---------- Internal methods ----------

    # ---------- Status Methods ----------

    def mark_dead(self) -> None:
        self.status = SessionStatus.DEAD

    def mark_bad(self, exception: Exception) -> None:
        self.status = SessionStatus.BAD
        self.bad_exception = exception

    @property
    def not_running(self) -> bool:
        return self.status != SessionStatus.RUNNING

    # ---------- Interfaces to ServeCore ----------

    async def add_request(self, request_payload: Dict) -> Dict:
        """Add a request to the session and assign a coroutine to the request.

        Args:
            request_payload (Dict): The request payload.

        Returns:
            Dict: The response payload.
        """

        # Convert the request to a RequestChain.
        chunked_request = ChunkedRequest.parse_from_payload(request_payload)
        request_chain = RequestChain.from_chunked_request(chunked_request)

        # Assign Semantic Variables to the RequestChain.
        self.var_mgr.create_vars(
            session_id=self.session_id, request_chain=request_chain
        )

        # Add the request to the executor.
        self.executor.add_request(request_chain=request_chain)

        # It must be inserted. So we can get the mapping.
        placeholder_mapping = request_chain.get_placeholder_mapping()

        return placeholder_mapping

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

    def exception_interrupt(self, exception: BaseException):
        self.status = SessionStatus.BAD
        self.bad_exception = exception

    def free_session(self):
        """Free the session and all its resources."""

        pass
