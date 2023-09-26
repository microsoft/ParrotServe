"""Executor is responsible for managing calls and scheduling to execute them."""

from typing import Dict, Optional
from abc import ABC, abstractmethod

from parrot.program.function import Prefix, LLMCall, Constant, ParameterLoc, ParamType
from parrot.program.future import Future
from parrot.orchestration.context import Context
from parrot.orchestration.controller import Controller
from parrot.orchestration.tokenize import TokenizedStorage
from parrot.utils import get_logger, create_task_in_loop

from .dispatcher import Dispatcher
from .session import Session
from .instructions import ConstantFill, PlaceholderFill, Generation
from .dataholder import DataHolder


logger = get_logger("Executor")


class BaseExecutor(ABC):
    """Base class for executors."""

    @abstractmethod
    def add_session(self, session: Session):
        pass


class NativeExecutor(BaseExecutor):
    """NativeExecutor for NativeBackends.

    NOTE(chaofan): Sessions under the same tokenizer are managed as a group."""

    def __init__(
        self,
        tokenizer_name: str,
        tokenized_storage: TokenizedStorage,
    ):
        # ---------- Resources ----------
        self.tokenizer_name = tokenizer_name
        self.tokenized_storage = tokenized_storage
        self.dataholder_map: Dict[int, DataHolder] = {}

    def add_session(self, session: Session):
        tokenized = self.tokenized_storage.tokenize_func_body(
            session.call.func,
            self.tokenizer_name,
        )

        eos_token_id = self.tokenized_storage.get_tokenizer(
            self.tokenizer_name
        ).eos_token_id

        # Translate function body to instructions
        for i, piece in enumerate(session.call.func.body):
            if isinstance(piece, Prefix):
                if (
                    session.call.func.cached_prefix
                    and session.call.shared_context_handler is None
                ):
                    continue  # If the prefix is cached, we do not need to fill it.
                inst = ConstantFill(tokenized[i])
            elif isinstance(piece, Constant):
                inst = ConstantFill(tokenized[i])
            elif isinstance(piece, ParameterLoc):
                assert piece.param.name in session.call.bindings
                param_value = session.call.bindings[piece.param.name]

                if piece.param.typ == ParamType.PYOBJ:
                    # For Python object, we directly fill the value.
                    # We use __str__ instead of __repr__
                    value_str = str(param_value)
                    inst = ConstantFill(
                        self.tokenized_storage.tokenize(
                            value_str,
                            self.tokenizer_name,
                        )
                    )
                else:
                    assert isinstance(param_value, Future)
                    holder = self._get_dataholder(param_value)
                    if piece.param.is_output:
                        assert param_value.is_middle_node
                        sampling_config = piece.param.sampling_config
                        # If not ignore_tokenizer_eos, we should add eos_token_id to stop_token_ids
                        if not sampling_config.ignore_tokenizer_eos:
                            sampling_config.stop_token_ids.append(eos_token_id)
                        inst = Generation(
                            output_holder=holder,
                            sampling_config=sampling_config,
                        )
                    else:
                        inst = PlaceholderFill(input_holder=holder)
            session.instructions.put_nowait(inst)

        create_task_in_loop(session.executing())

    def _get_dataholder(self, future: Future) -> DataHolder:
        # Create a new data future if not exists
        # Hence, the name of the future must be unique.
        if future.id not in self.dataholder_map:
            self.dataholder_map[future.id] = DataHolder(
                tokenizer=self.tokenizer_name,
                tokenized_storage=self.tokenized_storage,
                future=future,
            )
        return self.dataholder_map[future.id]


class MainExecutor:
    """Main executor is responsible for managing all the executors."""

    def __init__(self, controller: Controller, tokenized_storage: TokenizedStorage):
        # ---------- Global components ----------
        self.controller = controller
        self.controller.executor = self
        self.tokenized_storage = tokenized_storage

        # ---------- Dispatcher ----------
        self.dispatcher = Dispatcher(controller)

        # ---------- Sub-executors ----------
        # For NativeExectutor, it is: Tokenizer name -> NativeExecutor
        self.sub_executors: Dict[str, BaseExecutor] = {}

    def register_native_executor(self, tokenizer_name: str):
        self.sub_executors[tokenizer_name] = NativeExecutor(
            tokenizer_name,
            self.tokenized_storage,
        )

    def submit(self, call: LLMCall):
        new_created_context = True

        # Get/fork temporary context for a call
        if call.shared_context_handler is not None:
            if call.shared_context_handler.mode == "r":
                # Read mode: fork a new context
                context = Context(
                    parent_context=call.shared_context_handler.shared_context.context
                )
            else:
                # Write mode: directly use the context
                context = call.shared_context_handler.shared_context.context
                finish_callback = call.shared_context_handler.unlock_writer
                new_created_context = False
        elif call.func.cached_prefix:
            assert call.func.name in self.controller.function_prefix
            context = Context(
                parent_context=self.controller.function_prefix[call.func.name]
            )
        else:
            context = Context()

        if new_created_context:
            # If the context is a temporary one, we need to free it after execution;
            # Otherwise, we only unlock the writer of the context.
            finish_callback = context.destruction

        session = Session(call, context, finish_callback)
        self.dispatcher.dispatch(session)
        assert session.engine is not None

        if new_created_context:
            # And in this case, it must be a newly created context.
            # Hence we should set its cached_engines.
            context.cached_engines.add(session.engine)

        self.sub_executors[session.engine.tokenizer].add_session(session)

        logger.info(f"LLMCall {call.func.name} created a session {session.session_id}.")
