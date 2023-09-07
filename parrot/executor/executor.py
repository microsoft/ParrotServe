from typing import List, Dict
from queue import Queue
import time
import threading
import asyncio

from ..program.function import Promise, Constant, ParameterLoc
from ..program.placeholder import Placeholder
from .dispatcher import Dispatcher
from .session import Session
from .nodes import FillJob, GenerationJob, Tokensholder, JobStatus
from ..orchestration.context import Context
from ..orchestration.engine import ExecutionEngine
from ..orchestration.controller import Controller
from ..orchestration.tokenize import TokenizedStorage
from ..utils import get_logger
from ..protocol import free_context


log = get_logger("Executor")


class TokenizerGroupExecutor:
    """Sessions under the same tokenizer are managed as a group."""

    def __init__(
        self,
        tokenizer_name: str,
        tokenized_storage: TokenizedStorage,
    ):
        self.tokenizer_name = tokenizer_name
        self.tokenized_storage = tokenized_storage
        self.sessions: List[Session] = []

        # Placeholder name -> Tokensholder
        self.data_map: Dict[str, Tokensholder] = {}

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_daemon, daemon=True
        )
        self._execute_thread = threading.Thread(
            target=self._execute_daemon, daemon=True
        )
        self._execute_loop = asyncio.new_event_loop()

    def run_daemon(self):
        self._monitoring_thread.start()
        self._execute_thread.start()

    def add_session(self, session: Session):
        self.sessions.append(session)

        tokenized = self.tokenized_storage.tokenize_func_body(
            session.promise.func,
            self.tokenizer_name,
        )

        for i, piece in enumerate(session.promise.func.body):
            if isinstance(piece, Constant):
                holder = Tokensholder()
                holder.assign(tokenized[i])
                session.job_queue.append(
                    FillJob(status=JobStatus.WAITING, input_holder=holder)
                )
            elif isinstance(piece, ParameterLoc):
                assert piece.param.name in session.promise.bindings
                placeholder = session.promise.bindings[piece.param.name]
                holder = Tokensholder(
                    placeholder=placeholder, tokenized_storage=self.tokenized_storage
                )

                if piece.param.is_output:
                    session.job_queue.append(
                        GenerationJob(status=JobStatus.WAITING, output_holder=holder)
                    )
                else:
                    session.job_queue.append(
                        FillJob(status=JobStatus.WAITING, input_holder=holder)
                    )

        asyncio.run_coroutine_threadsafe(session.session_coro(), self._execute_loop)

    def _get_data_holder(self, placeholder: Placeholder) -> Tokensholder:
        if placeholder.name not in self.data_map:
            self.data_map[placeholder.name] = Tokensholder(
                placeholder=placeholder,
            )
        return self.data_map[placeholder.name]

    def _monitoring_daemon(self):
        while True:
            time.sleep(0.1)

            finished_sessions: List[Session] = []
            new_sessions: List[Session] = []
            for session in self.sessions:
                if session.finish_event.is_set():
                    finished_sessions.append(session)
                else:
                    new_sessions.append(session)
            self.sessions = new_sessions

    def _execute_daemon(self):
        self._execute_loop.run_forever()


class Executor:
    """Executor is responsible for managing promises and scheduling to
    execute them."""

    def __init__(self, controller: Controller, tokenized_storage: TokenizedStorage):
        self.controller = controller
        self.controller.executor = self
        self.dispatcher = Dispatcher(controller)
        self.tokenized_storage = tokenized_storage
        self.group_executors: Dict[str, TokenizerGroupExecutor] = {}

    def register_group_executor(self, tokenizer_name: str):
        self.group_executors[tokenizer_name] = TokenizerGroupExecutor(
            tokenizer_name,
            self.tokenized_storage,
        )

    def submit(self, promise: Promise):
        # Get/fork context
        if promise.func.name in self.controller.function_prefix:
            context = Context(self.controller.function_prefix[promise.func.name])
        else:
            context = Context()
        session = Session(promise, context)
        self.dispatcher.dispatch(session)
        self.group_executors[session.engine_name].add_session(session)

        log.info(f"Promise {promise.func.name} created a session {session.session_id}.")

    def run(self):
        for executor in self.group_executors.values():
            executor.run_daemon()
