from typing import List, Dict
from queue import Queue
import time
import threading
import asyncio

from ..program.function import Promise, Constant, ParameterLoc
from ..program.placeholder import Placeholder
from .dispatcher import Dispatcher
from .session import Session
from .nodes import FillJob, GenerationJob, Tokensholder
from ..orchestration.context import Context
from ..orchestration.engine import ExecutionEngine
from ..orchestration.controller import parrot_global_ctrl
from ..orchestration.tokenize import global_tokenized_storage, detokenize


class EngineExecutor:
    """Sessions under the same engine are managed as a group."""

    def __init__(self, engine: ExecutionEngine):
        self.engine: ExecutionEngine = engine
        self.sessions: List[Session] = []
        # Placeholder name -> Tokensholder
        self.data_map: Dict[str, Tokensholder] = {}
        self.detokenize_queue: Queue[Tokensholder] = Queue()

        self._detokenize_thread = threading.Thread(
            target=self._detokenize_daemon, daemon=True
        )
        self._execute_thread = threading.Thread(
            target=self._execute_daemon, daemon=True
        )
        self._execute_loop = asyncio.new_event_loop()

    def run_daemon(self):
        self._detokenize_thread.start()
        self._execute_thread.start()

    def add_session(self, session: Session):
        self.sessions.append(session)

        tokenized = global_tokenized_storage.tokenize_func_body(
            session.promise.func,
            self.engine.tokenizer,
        )

        for i, piece in enumerate(session.promise.func.body):
            if isinstance(piece, Constant):
                holder = Tokensholder()
                holder.assign(tokenized[i])
                session.job_queue.append(FillJob(input_holder=holder))
            elif isinstance(piece, ParameterLoc):
                holder = Tokensholder(placeholder=piece.param)
                if piece.param.is_output:
                    session.job_queue.append(GenerationJob(output_holder=holder))
                else:
                    session.job_queue.append(FillJob(input_holder=holder))

        asyncio.run_coroutine_threadsafe(session.session_coro(), self._execute_loop)

    def _get_data_holder(self, placeholder: Placeholder) -> Tokensholder:
        if placeholder.name not in self.data_map:
            self.data_map[placeholder.name] = Tokensholder(
                placeholder=placeholder,
            )
        return self.data_map[placeholder.name]

    def _detokenize_daemon(self):
        while True:
            time.sleep(0.1)
            if self.detokenize_queue.empty():
                continue
            holder = self.detokenize_queue.get()
            assert holder.ready
            # Push back to placeholder
            holder.placeholder.assign(
                detokenize(holder.token_ids, self.engine.tokenizer)
            )

    def _execute_daemon(self):
        self._execute_loop.run_forever()


class Executor:
    """Executor is responsible for managing promises and scheduling to
    execute them."""

    def __init__(self):
        self.dispatcher = Dispatcher()
        self.engine_executors: Dict[str, EngineExecutor] = {}
        for engine in parrot_global_ctrl.engines_table.values():
            self.engine_executors[engine.name] = EngineExecutor(engine)

    def submit(self, promise: Promise):
        # Get/fork context
        if promise.func.name in parrot_global_ctrl.function_prefix:
            context = Context(parrot_global_ctrl.function_prefix[promise.func.name])
        else:
            context = Context()
        session = Session(promise, context)
        self.dispatcher.dispatch(session)
        self.engine_executors[session.engine_name].add_session(session)

    def run(self):
        for executor in self.engine_executors.values():
            executor.run_daemon()


global_executor = Executor()
global_executor.run()
