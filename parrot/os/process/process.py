from typing import List, Dict, Optional
from queue import Queue

from parrot.program.future import Future
from parrot.program.function import SemanticCall
from parrot.utils import get_logger, RecyclePool
from parrot.constants import THREAD_POOL_SIZE
from parrot.exceptions import ParrotOSUserError

from .placeholder import Placeholder
from .thread import Thread
from .dag_node import DAGNode
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
        self.placeholders_map: Dict[int, Placeholder] = {}  # id -> placeholder

        # ---------- DAG ----------
        self.native_code_node = DAGNode()

        # ---------- Threads ----------
        self.threads: List[Thread] = []
        self.threads_pool = RecyclePool(THREAD_POOL_SIZE)

        # ---------- Runtime ----------
        self.dead = False  # Mark if the process is dead
        self.bad = False
        self.bad_exception: Optional[BaseException] = None
        self._calls: Queue[SemanticCall] = Queue()

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
        thread.engine.num_threads -= 1

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

    def _rewrite_call(self, call: SemanticCall):
        """Rewrite the futures to placeholders."""
        node = DAGNode(call)
        call.node = node

        meet_first_output = False

        for name, value in call.bindings.items():
            if not isinstance(value, Future):
                continue

            if value.id not in self.placeholders_map:
                self.placeholders_map[value.id] = Placeholder(value.id)

            call.bindings[name] = self.placeholders_map[value.id]

            # NOTE(chaofan): For simplicity, we only consider the edges before the first output.

            if call.func.params_map[name].is_input_loc and not meet_first_output:
                self.placeholders_map[value.id].out_nodes.append(node)
                node.add_in_edge()

            if call.func.params_map[name].is_output:
                meet_first_output = True

        for i, future in enumerate(call.output_futures):
            if future.id not in self.placeholders_map:
                self.placeholders_map[future.id] = Placeholder(value.id)
            call.output_futures[i] = self.placeholders_map[future.id]

    def _execute_call(self, call: SemanticCall):
        try:
            # Mark all placeholders as start
            for _, value in call.bindings.items():
                if isinstance(value, Placeholder):
                    value.start_event.set()

            # Get state context (if any)
            context_id = self.memory_space.get_state_context_id(
                pid=self.pid,
                func_name=call.func.name,
            )

            # Create a new thread
            thread = self._new_thread(call, context_id)

            # Dispatch the thread to some engine
            self.dispatcher.dispatch(thread)

            # Allocate memory
            self.memory_space.set_thread_ctx(thread)

            # Execute the thread
            self.executor.submit(thread)
        except ParrotOSUserError as e:
            self.exception_interrupt(e)

    # ---------- Interfaces to PCore ----------

    def exception_interrupt(self, exception: BaseException):
        self.bad = True
        self.bad_exception = exception

    def submit_call(self, call: SemanticCall):
        # Rewrite the call using namespace
        # Submit call will only put the call into a Queue, and the call will be executed later.
        # This is for get the partial DAG and do optimized scheduling.

        # NOTE(chaofan): For stateful call, since the queue is FIFO, the state contexts are
        # correctly maintained.
        self._rewrite_call(call)

        self._calls.put_nowait(call)

    def execute_calls(self):
        while not self._calls.empty():
            call = self._calls.get()
            self._execute_call(call)

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
