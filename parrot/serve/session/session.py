# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum
import asyncio
from typing import List, Dict, Optional
from queue import Queue

from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError, parrot_assert

from ..graph import SemanticVariable
from ..backend_repr import Context
from ..global_scheduler import TaskDispatcher
from ..tokenizer_wrapper import TokenizersWrapper
from ..context_manager import ServeCoreContextManager
from .graph_poller import GraphExecutor


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
        dispatcher: TaskDispatcher,
        context_mgr: ContextManager,
        tokenizers_wrapper: TokenizersWrapper,
    ):
        # ---------- Basic Info ----------
        self.session_id = session_id
        self.life_span = life_span  # In seconds

        # ---------- Global Components ----------
        self.context_mgr = context_mgr

        # ---------- Executor ----------
        self.executor = GraphExecutor(
            dispatcher=dispatcher,
            tokenizers_wrapper=tokenizers_wrapper,
        )

        # ---------- Runtime Status ----------
        self.status = SessionStatus.RUNNING
        self.bad_exception: Optional[Exception] = None

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

    def rewrite_call(self, call: BasicCall):
        r"""This function does two things:

        1. Rewrite the "Semantic Variables (Program-level)" to "Placeholders (OS-level)",
           using the namespace of the process.
        2. Make the DAG according to the dependencies between the placeholders.
        """

        # Rewrite SemanticVariable to Placeholder
        for name, value in call.bindings.items():
            if not isinstance(value, SemanticVariable):
                continue

            if value.id not in self.placeholders_map:
                self.placeholders_map[value.id] = SVPlaceholder(
                    id=value.id, name=value.name
                )

            call.bindings[name] = self.placeholders_map[value.id]
            if value.ready:
                self.placeholders_map[value.id].set(value.get())

        # Rewrite SemanticVariable to Placeholder
        for i, var in enumerate(call.output_vars):
            if var.id not in self.placeholders_map:
                self.placeholders_map[var.id] = SVPlaceholder(
                    id=value.id, name=value.name
                )
            call.output_vars[i] = self.placeholders_map[var.id]

        # Make DAG
        cur_edge = DAGEdge(call)
        call.edges.append(cur_edge)

        if isinstance(call, SemanticCall):
            """Semantic Call has multiple regions, hence multiple edges."""
            for region in call.func.body:
                call.edges_map[region.idx] = cur_edge

                if isinstance(region, ParameterLoc):
                    sv_placeholder = call.bindings[region.param.name]
                    if isinstance(sv_placeholder, SVPlaceholder):
                        if region.param.is_input_loc:
                            cur_edge.link_with_from_node(sv_placeholder)
                        elif region.param.is_output:
                            cur_edge.link_with_to_node(sv_placeholder)

                    if region.param.is_output:
                        # make a new node for the next segment
                        cur_edge = DAGEdge(call)
                        call.edges.append(cur_edge)

                        # Update max length of the placeholder
                        sv_placeholder.max_length = (
                            region.param.sampling_config.max_gen_length
                        )
        else:
            """Native Call has only one edge."""
            for name, value in call.bindings.items():
                if (
                    not isinstance(value, SVPlaceholder)
                    or call.func.params_map[name].is_output
                ):
                    continue

                sv_placeholder = self.placeholders_map[value.id]
                cur_edge.link_with_from_node(sv_placeholder)

            for var in call.output_vars:
                sv_placeholder = self.placeholders_map[var.id]
                cur_edge.link_with_to_node(sv_placeholder)

    def make_thread(self, call: SemanticCall) -> Thread:
        # Get state context (if any)
        # -1 indicates no state context
        context_id = self.memory_space.get_state_context_id(
            pid=self.pid,
            func_name=call.func.name,
        )

        call.thread = self._new_thread(call, context_id)
        return call.thread

    def execute_thread(self, thread: Thread):
        try:
            # Mark all placeholders as start
            for _, value in thread.call.bindings.items():
                if isinstance(value, SVPlaceholder):
                    value.start_event.set()

            # Allocate memory
            self.memory_space.set_thread_ctx(thread)

            # Execute the thread
            self.executor.submit(thread)
        except ParrotOSUserError as e:
            self.exception_interrupt(e)

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

    def free_process(self):
        self.monitor_threads()
        logger.info(
            f"Free process {self.pid} with running threads num: {len(self.threads)}"
        )
        self.memory_space.free_memory_space(self.pid)

    def monitor_threads(self):
        for thread in self.threads:
            if thread.finished:
                self._free_thread(thread)
