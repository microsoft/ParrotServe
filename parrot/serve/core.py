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

from parrot.serve.graph import PerformanceCriteria
from parrot.serve.scheduler import GlobalScheduler, GlobalSchedulerConfig

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

        self.engine_mgr = EngineManager(
            tokenizers_wrapper=self.tokenizers_wrapper,
            context_mgr=self.context_mgr,
            engine_heartbeat_timeout=self.config.engine_heartbeat_timeout,
        )

        self.session_mgr = SessionManager(
            life_span=self.config.session_life_span,
            prefix_matcher=self.prefix_matcher,
            scheduler=self.global_scheduler,
            var_mgr=self.var_mgr,
            engine_mgr=self.engine_mgr,
            context_mgr=self.context_mgr,
            tokenizers_wrapper=self.tokenizers_wrapper,
        )

        self.global_scheduler = GlobalScheduler(
            config=gs_config,
            engine_mgr=self.engine_mgr,
            context_mgr=self.context_mgr,
        )
        logger.info(
            f"PCore started with config: \n"
            + "\n".join(
                [f"  {key}={value}, " for key, value in self.os_config.__dict__.items()]
            )
        )

    # ---------- APIs to Engine Layer ----------

    def register_engine(self, config: EngineConfig) -> int:
        """Register a new engine in the OS.

        Args:
            config: EngineConfig. The engine config.

        Returns:
            int. The engine ID.
        """

        engine_id = self.engine_mgr.register_engine(config)
        return engine_id

    def engine_heartbeat(
        self,
        engine_id: int,
        engine_runtime_info: EngineRuntimeInfo,
    ) -> None:
        """Update the last seen time of an engine and other engine info.

        Args:
            engine_id: int. The engine ID.
            engine_runtime_info: EngineRuntimeInfo. The engine runtime info.
        """

        self.engine_mgr.engine_heartbeat(engine_id, engine_runtime_info)

    # ---------- Public Serving APIs ----------

    def register_session(self) -> int:
        """Register a new session in Serve Core.

        Returns:
            int: The session ID.
        """

        session_id = self.session_mgr.register_session()
        return session_id

    def remove_session(self, session_id: int) -> None:
        """Remove a session in Serve Core.

        Args:
            session_id: int. The session ID.
        """

        self.session_mgr.check_session_status(session_id)
        self.session_mgr._remove_session(session_id)

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

    def submit_semantic_call(self, request_payload: Dict) -> Dict:
        """Submit a semantic call in a session to the ServeCore.

        Args:
            request_payload: Dict. The remain request payload.

        Returns:
            A Dict. The response payload.
        """

        session_id = request_payload["session_id"]

        # The design of Parrot's completion API is asynchronous. We split up the "request"
        # into "submit" and "get" operations.
        # This is for get the partial DAG and do optimized scheduling.

        # Update session last access time
        self.session_mgr.check_session_status(session_id)
        self.session_mgr.session_access_update(session_id)

        # Add the request to the session.
        session = self.session_mgr.get_session(session_id)
        response = session.add_request(request_payload)
        response["session_id"] = session_id
        return response

    def semantic_variable_set(self, session_id: int, sv_id: str, content: str) -> None:
        """Set the content of a semantic variable.

        Args:
            session_id: int. The session ID.
            sv_id: str. The semantic variable ID.
            content: str. The content.
        """

        self.session_mgr.check_session_status(session_id)
        self.session_mgr.session_access_update(session_id)

        var = self.var_mgr.get_var(session_id, sv_id)
        var.set(content)

        logger.debug(
            f"SV set (id={sv_id}) from session (session_id={session_id}). "
            f"Set content length: {len(content)} "
        )

    async def semantic_variable_get(
        self, session_id: int, sv_id: str, criteria: str
    ) -> str:
        """Get the content from a Semantic Variable.

        Args:
            session_id: int. The session ID.
            sv_id: str. The Semantic Variable ID.
        """

        self.session_mgr.check_session_status(session_id)
        self.session_mgr.session_access_update(session_id)

        var = self.var_mgr.get_var(session_id, sv_id)

        if not var.activated:
            var.activate(criteria)

        await var.wait_ready()
        content = var.get()

        logger.debug(f"Semantic variable (id={sv_id}) get with criteria: {criteria}.")

        return content

    # ---------- ServeCore Loop ----------

    async def core_loop(self) -> None:
        """Start the Core loop."""

        while True:
            # Update and clean up sessions and engines
            self.session_mgr.check_running_sessions()
            self.session_mgr.sweep_not_running_sessions()
            self.engine_mgr.update_expired_engines()
            self.engine_mgr.sweep_not_running_engines()

            # Clean up expired constant prefix vars
            expired_vars = self.var_mgr.free_expired_constant_prefix_vars()
            for var in expired_vars:
                self.context_mgr.free_constant_prefix_contexts(var.sv_id)

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
