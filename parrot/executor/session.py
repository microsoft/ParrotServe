from typing import Optional
from queue import Queue

from .nodes import Job
from ..orchestration.context import Context
from ..program.function import Promise
from ..utils import RecyclePool

session_id_manager = RecyclePool(4096)


class Session:
    """A session represents a running promise in the executor."""

    def __init__(self, promise: Promise, context: Context):
        self.session_id = session_id_manager.allocate()
        self.promise = promise
        self.context = context
        # To be dispatched
        self.assigned_engine: Optional[str] = None
        self.job_queue: Queue[Job] = Queue()

    def __del__(self):
        session_id_manager.free(self.session_id)
