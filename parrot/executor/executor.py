from typing import List, Dict

from ..program.function import Promise
from .dispatcher import Dispatcher
from .session import Session
from .nodes import Job, FillJob, GenerationJob, Tokensholder
from ..orchestration.controller import parrot_global_ctrl
from ..orchestration.context import new_context


class EngineExecutor:
    """Sessions under the same engine are managed as a group."""

    def __init__(self):
        self.sessions: List[Session] = []
        self.running_jobs: List[Job] = []

    def add_session(self, session: Session):
        self.sessions.append(session)

        for piece in session.promise.func.body:
            pass


class Executor:
    """Executor is responsible for managing promises and scheduling to
    execute them."""

    def __init__(self):
        self.dispatcher = Dispatcher()
        self.engine_executors: Dict[str, EngineExecutor] = {}
        for tokenizer in parrot_global_ctrl.tokenizers_table:
            self.engine_executor[tokenizer] = EngineExecutor()

    def submit(self, promise: Promise):
        context = new_context()
        session = Session(promise, context)
        self.dispatcher.dispatch(session)
        self.engine_executors[session.assigned_engine].add_session(session)
