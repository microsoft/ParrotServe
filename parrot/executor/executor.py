from typing import Dict
import asyncio

from ..program.function import Promise, Constant, ParameterLoc
from ..program.placeholder import Placeholder
from .dispatcher import Dispatcher
from .session import Session
from .nodes import FillJob, GenerationJob, Tokensholder
from ..orchestration.context import Context
from ..orchestration.controller import Controller
from ..orchestration.tokenize import TokenizedStorage
from ..utils import get_logger


logger = get_logger("Executor")


class TokenizerGroupExecutor:
    """Sessions under the same tokenizer are managed as a group."""

    def __init__(
        self,
        tokenizer_name: str,
        tokenized_storage: TokenizedStorage,
    ):
        # ---------- Resources ----------
        self.tokenizer_name = tokenizer_name
        self.tokenized_storage = tokenized_storage
        self.data_map: Dict[str, Tokensholder] = {}

    def add_session(self, session: Session):
        tokenized = self.tokenized_storage.tokenize_func_body(
            session.promise.func,
            self.tokenizer_name,
        )

        for i, piece in enumerate(session.promise.func.body):
            if isinstance(piece, Constant):
                holder = Tokensholder(
                    tokenizer=self.tokenizer_name,
                    tokenized_storage=self.tokenized_storage,
                )
                holder.assign(tokenized[i])
                job = FillJob(input_holder=holder)
            elif isinstance(piece, ParameterLoc):
                assert piece.param.name in session.promise.bindings
                placeholder = session.promise.bindings[piece.param.name]
                holder = self._get_data_holder(placeholder)
                if piece.param.is_output:
                    job = GenerationJob(output_holder=holder)
                else:
                    job = FillJob(input_holder=holder)
            session.job_queue.put_nowait(job)

        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(session.session_coro(), loop)

    def _get_data_holder(self, placeholder: Placeholder) -> Tokensholder:
        if placeholder.name not in self.data_map:
            self.data_map[placeholder.name] = Tokensholder(
                tokenizer=self.tokenizer_name,
                tokenized_storage=self.tokenized_storage,
                placeholder=placeholder,
            )
        return self.data_map[placeholder.name]

    def _execute_daemon(self):
        self._execute_loop.run_forever()


class Executor:
    """Executor is responsible for managing promises and scheduling to
    execute them."""

    def __init__(self, controller: Controller, tokenized_storage: TokenizedStorage):
        # ---------- Global components ----------
        self.controller = controller
        self.controller.executor = self
        self.tokenized_storage = tokenized_storage

        # ---------- Dispatcher ----------
        self.dispatcher = Dispatcher(controller)

        # ---------- Group executors ----------
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
        self.group_executors[session.engine.tokenizer].add_session(session)

        logger.info(
            f"Promise {promise.func.name} created a session {session.session_id}."
        )
