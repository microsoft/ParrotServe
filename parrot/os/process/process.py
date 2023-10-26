from typing import List, Dict

from parrot.program.future import Future
from parrot.program.function import SemanticCall
from parrot.utils import get_logger, RecyclePool
from parrot.constants import THREAD_POOL_SIZE

from .placeholder import Placeholder
from .thread import Thread
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
        self.bad = False
        self.dead = False  # Mark if the process is dead

        # ---------- Threads ----------
        self.threads: List[Thread] = []
        self.threads_pool = RecyclePool(THREAD_POOL_SIZE)

    @property
    def live(self):
        return not self.dead and not self.bad

    def _new_thread(self, call: SemanticCall) -> Thread:
        tid = self.threads_pool.allocate()
        thread = Thread(tid=tid, process=self, call=call)
        self.threads.append(thread)
        return thread

    def _free_thread(self, thread: Thread):
        logger.info(f"Free thread {thread.tid}")
        self.threads_pool.free(thread.tid)
        self.threads.remove(thread)
        self.memory_space.free_thread_memory(thread)

    def _rewrite_call(self, call: SemanticCall):
        """Rewrite the futures to placeholders."""
        for name, value in call.bindings.items():
            if not isinstance(value, Future):
                continue

            if value.id not in self.placeholders_map:
                self.placeholders_map[value.id] = Placeholder(value.id)

            call.bindings[name] = self.placeholders_map[value.id]

        for i, future in enumerate(call.output_futures):
            if future.id not in self.placeholders_map:
                self.placeholders_map[future.id] = Placeholder(value.id)
            call.output_futures[i] = self.placeholders_map[future.id]

    def _execute_call(self, call: SemanticCall):
        # Rewrite the call using namespace
        self._rewrite_call(call)

        # Create a new thread
        thread = self._new_thread(call)

        # Dispatch the thread to some engine
        self.dispatcher.dispatch(thread)

        # Allocate memory
        self.memory_space.set_thread_ctx(thread)

        # Execute the thread
        self.executor.submit(thread)

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
