# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import json
from typing import Dict
import asyncio

from parrot.utils import get_logger
from parrot.constants import CORE_LOOP_INTERVAL
from parrot.protocol.internal.runtime_info import EngineRuntimeInfo
from parrot.engine.config import EngineConfig
from parrot.exceptions import ParrotCoreInternalError

from parrot.serve.graph import (
    PlaceholderGen,
    get_performance_criteria,
    activate_producer,
)
from parrot.serve.scheduler import GlobalScheduler, GlobalSchedulerConfig, TaskCreator

from .config import ServeCoreConfig
from .prefix_matcher import PrefixMatcher
from .variable_manager import SemanticVariableManager
from .tokenizer_wrapper import TokenizersWrapper
from .context_manager import ServeCoreContextManager
from .session_manager import SessionManager
from .engine_manager import EngineManager


logger = get_logger("ServeCore")


class ParrotServeCore:
    """ServeCore is a central manager for the Parrot Serve layer.

    It serves requests from the frontend, in the form of Parrot's standard API (Completion API w/ Semantic Variables.).

    It connects:
    - Multiple sessions in the frontend.
    - Multiple engines in the backend, attached with different models.

    It manages:
    - Contexts.
    - Semantic variables.
    - Tokenizer.

    There is a GlobalScheduler to schedule and dispatch Tasks generated from sessions' GraphExecutor
    to different engines.
    """

    def __init__(self, config: Dict):
        # ---------- Config ----------
        gs_config = config.pop("global_scheduler")
        gs_config = GlobalSchedulerConfig(**gs_config)
        self.config = ServeCoreConfig(**config)

        # ---------- Components ----------
        self.prefix_matcher = PrefixMatcher()
        self.var_mgr = SemanticVariableManager(
            constant_prefix_var_timeout=self.config.constant_prefix_var_timeout
        )
        self.tokenizers_wrapper = TokenizersWrapper()
        self.context_mgr = ServeCoreContextManager()
        self.task_creator = TaskCreator()

        self.engine_mgr = EngineManager(
            tokenizers_wrapper=self.tokenizers_wrapper,
            context_mgr=self.context_mgr,
            engine_heartbeat_timeout=self.config.engine_heartbeat_timeout,
        )

        self.global_scheduler = GlobalScheduler(
            config=gs_config,
            engine_mgr=self.engine_mgr,
            context_mgr=self.context_mgr,
        )

        self.session_mgr = SessionManager(
            life_span=self.config.session_life_span,
            prefix_matcher=self.prefix_matcher,
            task_creator=self.task_creator,
            scheduler=self.global_scheduler,
            var_mgr=self.var_mgr,
            engine_mgr=self.engine_mgr,
            context_mgr=self.context_mgr,
            tokenizers_wrapper=self.tokenizers_wrapper,
        )

        logger.info(
            f"Parrot ServeCore started with config: \n"
            + "\n".join(
                [f"  {key}={value}, " for key, value in self.config.__dict__.items()]
            )
        )

    # ---------- APIs to Engine Layer ----------

    def register_engine(self, payload: Dict) -> Dict:
        """Register a new engine in the OS.

        Args:
            config: EngineConfig. The engine config.

        Returns:
            Dict. The response.
        """

        logger.debug(f"Register engine received.")
        engine_config = EngineConfig(**payload["engine_config"])
        engine_id = self.engine_mgr.register_engine(engine_config)
        return {"engine_id": engine_id}

    def engine_heartbeat(self, payload: Dict) -> Dict:
        """Update the last seen time of an engine and other engine info.

        Args:
            engine_id: int. The engine ID.
            engine_runtime_info: EngineRuntimeInfo. The engine runtime info.

        Returns:
            Dict. The response.
        """

        engine_id = payload["engine_id"]
        engine_name = payload["engine_name"]
        logger.debug(f"Engine {engine_name} (id={engine_id}) heartbeat received.")
        engine_info = EngineRuntimeInfo(**payload["runtime_info"])

        self.engine_mgr.engine_heartbeat(engine_id, engine_info)

        return {}

    # ---------- Public Serving APIs ----------

    # ---------- Session Management ----------

    def register_session(self, payload: Dict) -> Dict:
        """Register a new session in Serve Core.

        Args:
            payload: Dict. The payload.

        Returns:
            Dict. The response.
        """

        session_id = self.session_mgr.register_session()
        return {"session_id": session_id, "session_auth": "1"}

    def remove_session(self, session_id: int, payload: Dict) -> Dict:
        """Remove a session in Serve Core.

        Args:
            session_id: int. The session ID.
            payload: Dict. The payload.

        Returns:
            Dict. The response.
        """

        self.session_mgr.check_session_status(session_id)
        self.session_mgr.remove_session(session_id)

        return {}

    def get_session_info(self, session_id: int, payload: Dict) -> Dict:
        """Get the session info.

        Args:
            session_id: int. The session ID.
            payload: Dict. The payload.

        Returns:
            Dict. The response.
        """

        return {}

    # ---------- Function Call ----------

    # TODO: Support native call
    # async def submit_native_call(self, pid: int, call: NativeCall) -> int:
    #     """Submit a native call from a VM to the OS."""

    #     # The native call must be a short, executable and stateless call. (FaaS)
    #     # The native call will be executed immediately once all its inputs are ready.

    #     self._check_process(pid)
    #     process = self.processes[pid]

    #     # Rewrite the call using namespace
    #     process.rewrite_call(call)

    #     # Execute it immediately
    #     process.execute_native_call(call)

    def submit_semantic_call(self, payload: Dict) -> Dict:
        """Submit a semantic call in a session to the ServeCore.

        Args:
            payload: Dict. The request payload.

        Returns:
            Dict. The response.
        """

        session_id = payload["session_id"]

        # The design of Parrot's completion API is asynchronous. We split up the "request"
        # into "submit" and "get" operations.
        # This is for get the partial DAG and do optimized scheduling.

        # Update session last access time
        self.session_mgr.check_session_status(session_id)
        self.session_mgr.session_access_update(session_id)

        # Add the request to the session.
        session = self.session_mgr.get_session(session_id)
        request_id, created_vars = session.add_request(payload)

        return {
            "request_id": request_id,
            "created_vars": created_vars,
        }

    # ---------- Semantic Variable ----------

    def register_semantic_variable(self, payload: Dict) -> Dict:
        """Register a semantic variable in a session.

        Args:
            payload: Dict. The payload.

        Returns:
            Dict. The response.
        """

        session_id = payload["session_id"]
        name = payload["var_name"]

        self.session_mgr.check_session_status(session_id)
        self.session_mgr.session_access_update(session_id)

        var = self.var_mgr.create_var(session_id, name)
        logger.debug(
            f"SV registered (id={var.id}) in session (session_id={session_id})."
        )

        return {"var_id": var.id}

    def set_semantic_variable(self, var_id: str, payload: Dict) -> Dict:
        """Set the content of a semantic variable.

        Args:
            var_id: str. The variable ID.
            payload: Dict. The payload.

        Returns:
            Dict. The response.
        """

        session_id = payload["session_id"]
        content = payload["content"]

        self.session_mgr.check_session_status(session_id)
        self.session_mgr.session_access_update(session_id)

        var = self.var_mgr.get_var(session_id, var_id)
        var.set(content)

        logger.debug(
            f"SV set (id={var_id}) from session (session_id={session_id}). "
            f"Set content length: {len(content)} "
        )

        return {}

    async def get_semantic_variable(self, var_id: str, payload: Dict) -> Dict:
        """Get the content from a Semantic Variable.

        Args:
            var_id: str. The variable ID.
            payload: Dict. The payload.

        Returns:
            Dict. The response.
        """

        session_id = payload["session_id"]
        criteria = payload["criteria"]

        self.session_mgr.check_session_status(session_id)
        self.session_mgr.session_access_update(session_id)

        var = self.var_mgr.get_var(session_id, var_id)
        if var.has_producer:
            producer: PlaceholderGen = var.get_producer()
            if not producer.comp_chain.is_activated:
                # Activate the chain and propagate the performance criteria
                activate_producer(
                    producer.comp_chain, get_performance_criteria(criteria)
                )

        await var.wait_ready()
        content = var.get()

        logger.debug(f"Semantic variable (id={var_id}) get with criteria: {criteria}.")

        return {"content": content}

    # ---------- ServeCore Loop ----------

    async def serve_loop(self) -> None:
        """Start the Core serving loop."""

        while True:
            # Update and clean up sessions and engines
            self.session_mgr.check_running_sessions()
            self.session_mgr.sweep_not_running_sessions()
            self.engine_mgr.update_expired_engines()
            self.engine_mgr.sweep_not_running_engines()

            # Clean up expired constant prefix vars
            expired_vars = self.var_mgr.free_expired_constant_prefix_vars()
            for var in expired_vars:
                self.context_mgr.free_constant_prefix_contexts(var.id)

            # Schedule tasks
            self.global_scheduler.schedule()

            await asyncio.sleep(CORE_LOOP_INTERVAL)


def create_serve_core(
    core_config_path: str,
    release_mode: bool = False,
    override_args: Dict = {},
) -> ParrotServeCore:
    """Create the ServeCore.

    Args:
        core_config_path: str. The path to the ServeCore config file.
        release_mode: bool. Whether to run in release mode.
        override_args: Dict. The override arguments.

    Returns:
        ParrotServeCore. The created Parrot Serve Core.
    """

    with open(core_config_path) as f:
        core_config = dict(json.load(f))

    core_config.update(override_args)

    if not ServeCoreConfig.verify_config(core_config):
        raise ParrotCoreInternalError(f"Invalid ServeCore config: {core_config}")

    return ParrotServeCore(core_config)
