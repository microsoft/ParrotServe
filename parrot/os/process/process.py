from typing import List

from parrot.program.function import SemanticCall
from parrot.utils import get_logger, RecyclePool
from parrot.constants import THREAD_POOL_SIZE

from .thread import Thread
from .thread_dispatcher import ThreadDispatcher
from ..memory.mem_space import MemorySpace
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
        self, pid: int, dispatcher: ThreadDispatcher, memory_space: MemorySpace
    ):
        self.pid = pid
        self.dispatcher = dispatcher
        self.memory_space = memory_space
        self.executor = Executor(self)

        # ---------- Threads ----------
        self.threads: List[Thread] = []
        self.threads_pool = RecyclePool(THREAD_POOL_SIZE)

    def _new_thread(self, call: SemanticCall):
        tid = self.threads_pool.allocate()
        thread = Thread(tid, self, call)
        self.threads.append(thread)
        return thread

    def _free_thread(self, thread: Thread):
        self.threads_pool.free(thread.tid)
        self.threads.remove(thread)

    def execute_call(self, call: SemanticCall):
        thread = self._new_thread(call)
        self.dispatcher.dispatch(thread)
