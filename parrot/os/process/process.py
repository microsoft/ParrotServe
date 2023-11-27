# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Dict, Optional
from queue import Queue

from parrot.program.semantic_variable import ParameterLoc
from parrot.program.future import Future
from parrot.program.function import SemanticCall
from parrot.utils import get_logger, RecyclePool
from parrot.constants import THREAD_POOL_SIZE
from parrot.exceptions import ParrotOSUserError

from .placeholder import SVPlaceholder
from .thread import Thread
from .dag_edge import DAGEdge
from ..thread_dispatcher import ThreadDispatcher
from ..memory.mem_space import MemorySpace
from ..tokenizer import Tokenizer
from .executor import Executor


logger = get_logger("Process")


class Process:
    """
    The process is an abstraction for the VM to interact with the OS: When a VM connected to the OS,
    a process will be created for it.

    A process has its own executor and tokenizer.
    But the memory space (context) and thread dispatcher are shared between processes.
    """

    def __init__(
        self,
        pid: int,
        dispatcher: ThreadDispatcher,
        memory_space: MemorySpace,
        tokenizer: Tokenizer,
    ):
        # ---------- Basic Info ----------
        self.pid = pid
        self.dispatcher = dispatcher
        self.memory_space = memory_space
        self.memory_space.new_memory_space(self.pid)
        self.executor = Executor(tokenizer)
        self.placeholders_map: Dict[int, SVPlaceholder] = {}  # id -> placeholder

        # ---------- Threads ----------
        self.threads: List[Thread] = []
        self.threads_pool = RecyclePool(THREAD_POOL_SIZE)

        # ---------- Runtime ----------
        self.dead = False  # Mark if the process is dead
        self.bad = False
        self.bad_exception: Optional[BaseException] = None

    @property
    def live(self):
        return not self.dead and not self.bad

    # ---------- Internal ----------

    def _new_thread(self, call: SemanticCall, context_id: int) -> Thread:
        tid = self.threads_pool.allocate()
        thread = Thread(tid=tid, process=self, call=call, context_id=context_id)
        self.threads.append(thread)
        return thread

    def _free_thread(self, thread: Thread):
        logger.info(f"Free thread {thread.tid}")
        self.threads_pool.free(thread.tid)
        self.threads.remove(thread)

        thread.engine.remove_thread(thread)

        # For stateful call
        if not thread.is_stateful:
            self.memory_space.free_thread_memory(thread)
        else:
            # Maintain the stateful context
            self.memory_space.set_state_context_id(
                pid=self.pid,
                func_name=thread.call.func.name,
                context_id=thread.context_id,
            )

    # ---------- Interfaces to PCore ----------

    def rewrite_call(self, call: SemanticCall):
        r"""This function does two things:

        1. Rewrite the "Futures (Program-level)" to "Placeholders (OS-level)", using the namespace of the process.
        2. Make the DAG according to the dependencies between the placeholders.
        """

        # Rewrite future to Placeholder
        for name, value in call.bindings.items():
            if not isinstance(value, Future):
                continue

            if value.id not in self.placeholders_map:
                self.placeholders_map[value.id] = SVPlaceholder(
                    id=value.id, name=value.name
                )

            call.bindings[name] = self.placeholders_map[value.id]

        for i, future in enumerate(call.output_futures):
            if future.id not in self.placeholders_map:
                self.placeholders_map[future.id] = SVPlaceholder(
                    id=value.id, name=value.name, producer=call
                )
            call.output_futures[i] = self.placeholders_map[future.id]

        # Make DAG
        cur_edge = DAGEdge(call)
        call.edges.append(cur_edge)

        for sv in call.func.body:
            call.edges_map[sv.idx] = cur_edge

            if isinstance(sv, ParameterLoc):
                sv_placeholder = call.bindings[sv.param.name]
                if isinstance(sv_placeholder, SVPlaceholder):
                    if sv.param.is_input_loc:
                        cur_edge.link_with_in_node(sv_placeholder)
                    elif sv.param.is_output:
                        cur_edge.link_with_out_node(sv_placeholder)

                if sv.param.is_output:
                    # make a new node for the next segment
                    cur_edge = DAGEdge(call)
                    call.edges.append(cur_edge)

    def make_thread(self, call: SemanticCall) -> Thread:
        # Get state context (if any)
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

    def exception_interrupt(self, exception: BaseException):
        self.bad = True
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
