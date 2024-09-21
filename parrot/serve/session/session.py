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
    PyNativeCallRequest,
    RequestChain,
    NativeFuncNode,
    get_performance_criteria,
    activate_sv,
)

from parrot.serve.backend_repr import Context
from parrot.serve.scheduler import TaskCreator, GlobalScheduler

from ..prefix_matcher import PrefixMatcher
from ..variable_manager import SemanticVariableManager
from ..engine_manager import EngineManager
from ..tokenizer_wrapper import TokenizersWrapper
from ..context_manager import ServeCoreContextManager
from .graph_executor import GraphExecutor
from .native_executor import PyNativeExecutor


logger = get_logger("Session")


class SessionStatus(Enum):
    RUNNING = 0  # The session is running
    DEAD = 1  # The session is dead, i.e. the program (in the frontend) is disconnected / timed out
    BAD = 2  # The session is bad, i.e. the session throws an exception during execution


class Session:
    """
    A session is an abstraction of a program interacting with the ServeCore: When a program connected to the ServeCore,
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
        self.prefix_matcher = prefix_matcher
        self.var_mgr = var_mgr
        self.context_mgr = context_mgr

        # ---------- Executor ----------
        self.executor = GraphExecutor(
            session_id=session_id,
            task_creator=task_creator,
            scheduler=scheduler,
            engine_mgr=engine_mgr,
            context_mgr=context_mgr,
            tokenizers_wrapper=tokenizers_wrapper,
        )
        self.native_executor = PyNativeExecutor(
            session_id=session_id, graph=self.executor.graph
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

    def _add_semantic_request(self, request_id: int, request_payload: Dict) -> List:
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
        self.var_mgr.create_vars_for_semantic_request_chain(
            session_id=self.session_id, request_chain=request_chain
        )

        # Add the request to the executor.
        self.executor.add_request(request_chain=request_chain)

        # If criteria is specified, activate it immediately.
        if chunked_request.metadata.output_criteria is not None:
            criteria = chunked_request.metadata.output_criteria
            if isinstance(criteria, str):
                criteria = get_performance_criteria(criteria)
            activate_sv(request_chain.comp_chains[-1].gen_node.sv, criteria)

        # It must be inserted. So we can get the mapping.
        param_info = request_chain.get_param_info()

        return param_info

    def _add_py_native_request(self, request_id: int, request_payload: Dict) -> List:
        # Convert the request to a PyNativeCallRequest.
        request = PyNativeCallRequest.parse_from_payload(
            request_id=request_id,
            session_id=self.session_id,
            payload=request_payload,
        )

        # Convert the PyNativeCallRequest to a FuncNode.
        func_node = NativeFuncNode(native_func=request)

        # Assign Semantic Variables to the Func Node.
        self.var_mgr.create_vars_for_pynative_func(func_node)

        # Add the request to the executor.
        self.native_executor.add_native_func(func_node)

        # If criteria is specified, activate it immediately.
        if request.metadata.output_criteria is not None:
            criteria = request.metadata.output_criteria
            if isinstance(criteria, str):
                criteria = get_performance_criteria(criteria)
            for out_var in func_node.output_vars.values():
                activate_sv(out_var, criteria)

        # It must be inserted. So we can get the mapping.
        param_info = func_node.get_param_info()

        return param_info

    # ---------- Interfaces to ServeCore ----------

    def add_request(self, request_payload: Dict, is_native: bool) -> (int, List):
        """Add a request to the session and assign a coroutine to the request.

        Args:
            request_payload (Dict): The request payload.

        Returns:
            int: The request id.
            List: The param info.
        """

        # Get the request id.
        request_id = self._request_id_counter
        self._request_id_counter += 1

        if is_native:
            param_info = self._add_py_native_request(request_id, request_payload)
        else:
            param_info = self._add_semantic_request(request_id, request_payload)

        return request_id, param_info

    def _register_session_resources(self) -> None:
        self.context_mgr.register_session_contexts(session_id=self.session_id)
        self.var_mgr.register_local_var_space(session_id=self.session_id)

    def free_session_resources(self) -> None:
        """Free the session and all its resources."""

        # Free the contexts of session.
        self.context_mgr.free_session_contexts(session_id=self.session_id)

        # Free the local var space of the session.
        self.var_mgr.free_local_var_space(session_id=self.session_id)

        logger.info(
            f"Free Session(session_id={self.session_id}) with running tasks num: {0}"
        )
