from parrot.utils import get_logger
from .process.thread import Thread


logger = get_logger("ThreadDispatcher")


class ThreadDispatcher:
    """ThreadDispatcher, or called ThreadScheduler, is responsible for dispatching threads
    to different backend engines.

    It is shared between different processes, so that it can has the global view of all threads
    from different processes. For exmaple, threads from different processes with the same prefix
    can be scheduled to the same engine.
    """

    def __init__(self, os: "PCore"):
        self.os = os

    def dispatch(self, thread: Thread):
        """Dispatch a thread to some backend engine."""

        # TODO: Implement the dispatching strategy.
        thread.engine = self.os.engines.values()[0]

        logger.info(f"Thread {thread.tid} dispatched to engine {thread.engine.name}.")
