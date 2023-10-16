from typing import Optional, Literal

from Parrot.parrot.os.engine import ExecutionEngine
from Parrot.parrot.os.memory.context import Context
from parrot.protocol import fill

from .function import SemanticFunction, SemanticCall, logger


class SharedContext:
    """Shared context for different functions.

    A function call (SemanticCall) can specify a shared context to read-only/read-write.
    Usage:
    - ctx = P.shared_context(engine, init_text)
    - ctx.fill(text)
    - with ctx.open(mode="r") as h:
        h.call(func_name, func_args, func_kwargs)
    """

    _controller: Optional["Controller"] = None
    _tokenized_storage: Optional["TokenizedStorage"] = None

    def __init__(
        self,
        engine_name: str,
        parent_context: Optional[Context],
    ):
        if self._controller is None:
            raise RuntimeError("Controller for class `SharedContext` is not set.")

        if engine_name not in self._controller.engines_table:
            raise ValueError(f"Engine {engine_name} is not registered.")

        self.engine: ExecutionEngine = self._controller.engines_table[engine_name]
        self.context = Context(parent_context=parent_context)

        self._free_flag = False

    def __del__(self):
        if not self._free_flag:
            self.context.destruction()

    def fill(self, text: str):
        self._check_free()
        if not self._controller.is_running:
            raise RuntimeError(
                "Controller is not running. "
                "(Please ensure call `vm.init` before running a Parrot function."
            )

        if self._tokenized_storage is None:
            raise RuntimeError(
                "Tokenized storage for class `SharedContext` is not set."
            )

        encoded = self._tokenized_storage.tokenize(text, self.engine.tokenizer)

        self.context.cached_engines.add(self.engine)
        resp = fill(
            http_addr=self.engine.http_address,
            client_id=self.engine.client_id,
            context_id=self.context.context_id,
            parent_context_id=self.context.parent_context_id,
            token_ids=encoded,
        )
        assert resp.num_filled_tokens == len(
            encoded
        ), "Shared context filled failed: not all tokens are filled."

    def open(self, mode: Literal["r", "w"]):
        self._check_free()
        return SharedContextHandler(self, mode)

    def _check_free(self):
        if self._free_flag:
            raise RuntimeError("Shared context has been freed.")

    def free(self):
        self._check_free()
        self.context.destruction()
        self._free_flag = True


class SharedContextHandler:
    """SharedContextHandler indicates a open shared context."""

    def __init__(self, shared_context: SharedContext, mode: str):
        self.shared_context = shared_context
        self.mode = mode

        self._writer: Optional[SemanticCall] = None  # Record current writer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def unlock_writer(self):
        self._writer = None

    def call(self, func: SemanticFunction, *args, **kwargs):
        call = SemanticCall(func, self, *args, **kwargs)

        if self._writer is not None:
            raise RuntimeError(
                "Conflict: currently this shared context is being written. Any read/write is not allowed."
            )

        if self.mode == "w":
            self._writer = call

        if SemanticFunction._executor is not None:
            SemanticFunction._executor.submit(call)
        else:
            logger.warning(
                "Executor is not set, will not submit the call. "
                "(Please ensure run a Parrot function in a running context, e.g. using env.parrot_run_aysnc)"
            )

        # Unpack the output futures
        if len(call.output_futures) == 1:
            return call.output_futures[0]
        return tuple(call.output_futures)
