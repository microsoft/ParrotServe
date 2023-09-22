from typing import Dict

from ..program.function import Prefix, Promise, Constant, ParameterLoc, ParamType
from ..program.placeholder import Placeholder
from .dispatcher import Dispatcher
from .session import Session
from .instructions import ConstantFill, PlaceholderFill, Generation
from .tokens_holder import TokensHolder
from ..orchestration.context import Context
from ..orchestration.controller import Controller
from ..orchestration.tokenize import TokenizedStorage
from ..utils import get_logger, create_task_in_loop


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
        self.tokensholder_map: Dict[str, TokensHolder] = {}

    def add_session(self, session: Session):
        tokenized = self.tokenized_storage.tokenize_func_body(
            session.promise.func,
            self.tokenizer_name,
        )

        # Hack: we append eos_token_id in the sampling params
        # It should be iterated in the future because the sampling params should be unique for every generation.
        eos_token_id = self.tokenized_storage.get_tokenizer(
            self.tokenizer_name
        ).eos_token_id
        session.sampling_params.stop_token_ids.append(eos_token_id)

        # Translate function body to instructions
        for i, piece in enumerate(session.promise.func.body):
            if isinstance(piece, Prefix):
                if session.promise.func.cached_prefix:
                    continue  # If the prefix is cached, we do not need to fill it.
                inst = ConstantFill(tokenized[i])
            elif isinstance(piece, Constant):
                inst = ConstantFill(tokenized[i])
            elif isinstance(piece, ParameterLoc):
                assert piece.param.name in session.promise.bindings
                param_value = session.promise.bindings[piece.param.name]
                if piece.param.typ == ParamType.PYOBJ:
                    # We use __str__ instead of __repr__
                    value_str = str(param_value)
                    inst = ConstantFill(
                        self.tokenized_storage.tokenize(
                            value_str,
                            self.tokenizer_name,
                        )
                    )
                else:
                    assert isinstance(param_value, Placeholder)
                    holder = self._get_data_holder(param_value)
                    if piece.param.is_output:
                        inst = Generation(output_holder=holder)
                    else:
                        inst = PlaceholderFill(input_holder=holder)
            session.instructions.put_nowait(inst)

        create_task_in_loop(session.execute_coroutine())

    def _get_data_holder(self, placeholder: Placeholder) -> TokensHolder:
        # Create a new data holder if not exists
        # Hence, the name of the placeholder must be unique.
        if placeholder.name not in self.tokensholder_map:
            self.tokensholder_map[placeholder.name] = TokensHolder(
                tokenizer=self.tokenizer_name,
                tokenized_storage=self.tokenized_storage,
                placeholder=placeholder,
            )
        return self.tokensholder_map[placeholder.name]


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
        if promise.func.cached_prefix:
            assert promise.func.name in self.controller.function_prefix
            context = Context(
                parent_context=self.controller.function_prefix[promise.func.name]
            )
        else:
            context = Context()

        session = Session(promise, context)
        self.dispatcher.dispatch(session)
        self.group_executors[session.engine.tokenizer].add_session(session)

        logger.info(
            f"Promise {promise.func.name} created a session {session.session_id}."
        )
