import asyncio
import contextlib
import time
import json
from typing import Optional

from .orchestration.controller import Controller
from .executor.executor import Executor
from .program.function import SemanticFunction
from .program.shared_context import SharedContext
from .orchestration.tokenize import TokenizedStorage


class VirtualMachine:
    """The virtual machine for Parrot."""

    def __init__(self, config_path: Optional[str] = None):
        """The config file contains pre-registered tokenizers and engines."""

        if config_path is None:
            self.config = None
        else:
            with open(config_path) as f:
                self.config = json.load(f)

    def init(self):
        """When you start definiing a Parrot function/shared_context,
        you should init the VirtualMachine first."""

        self.controller = Controller()
        self.tokenized_storage = TokenizedStorage(self.controller)
        self.executor = Executor(self.controller, self.tokenized_storage)

        SemanticFunction._controller = self.controller
        SharedContext._controller = self.controller
        SharedContext._tokenized_storage = self.tokenized_storage

        if self.config is not None:
            if "mode" in self.config and self.config["mode"] == "release":
                import logging

                # We don't disable the error log
                logging.disable(logging.DEBUG)
                logging.disable(logging.INFO)
            if "tokenizers" in self.config:
                for tokenizer_name in self.config["tokenizers"]:
                    self.register_tokenizer(tokenizer_name)
            if "engines" in self.config:
                for engine_info in self.config["engines"]:
                    self.register_engine(
                        **engine_info,
                    )

    @contextlib.contextmanager
    def running_scope(self, timeit: bool = False):
        """Under this context, the global controller is running."""

        # Set the executor
        SemanticFunction._executor = self.executor

        self.controller.run()
        self.controller.caching_function_prefix(self.tokenized_storage)

        if timeit:
            st = time.perf_counter_ns()

        try:
            yield
        except BaseException as e:
            # This is mainly used to catch the error in the `main`
            #
            # For errors in coroutines, we use the fail fast mode and quit the whole system
            # In this case, we can only see a SystemExit error
            print("Error happens when executing Parrot program: ", type(e), repr(e))
            # print("Traceback: ", traceback.format_exc())
        else:
            if timeit:
                ed = time.perf_counter_ns()
                print(f"[Timeit] E2E Program Execution Time: {(ed - st) / 1e9} (s).")

            self.controller.free_function_prefix()

    def register_tokenizer(self, tokenizer_name: str):
        """Register a tokenizer to the global controller."""

        self.controller.register_tokenizer(tokenizer_name)

    def register_engine(
        self,
        engine_name: str,
        host: str,
        port: int,
        tokenizer: str,
    ):
        """Register an engine to the global controller."""

        self.controller.register_engine(engine_name, host, port, tokenizer)

    def run(self, coroutine, timeit: bool = False):
        """vm.run method will create a new event loop and run the coroutine."""

        with self.running_scope(timeit):
            # asyncio.run(coroutine)
            loop = asyncio.new_event_loop()
            loop.run_until_complete(coroutine)
            loop.close()
