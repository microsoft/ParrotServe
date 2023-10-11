import asyncio
import contextlib
import time
import traceback
from typing import Coroutine, Literal

from parrot.protocol.layer_apis import register_vm
from parrot.program.function import SemanticFunction
from parrot.program.shared_context import SharedContext
from parrot.utils import get_logger

from .controller import Controller
from .executor import Executor
from .tokenizer import Tokenizer


logger = get_logger("VM")


class VirtualMachine:
    """The virtual machine for Parrot."""

    def __init__(self, os_http_addr):
        self.os_http_addr = os_http_addr

        # Call OS to register VM, and allocate a pid
        resp = register_vm(self.os_http_addr)
        self.pid = resp.pid

        logger.info(f"VM created with PID: {self.pid}")

    def launch(self, mode: Literal["release", "debug"] = "release"):
        """Run the global controller and take over the control of the whole system (Change some class members.).

        When you start definiing a Parrot function/shared_context, you should init the VirtualMachine
        first."""

        self.controller = Controller(self)
        self.tokenizer = Tokenizer()
        self.executor = Executor(self, self.tokenizer)

        SemanticFunction._controller = self.controller
        SharedContext._controller = self.controller
        SharedContext._tokenized_storage = self.tokenizer

        if mode == "release":
            import logging

            # We don't disable the error log
            logging.disable(logging.DEBUG)
            logging.disable(logging.INFO)

        # Set the executor
        SemanticFunction._executor = self.executor

        self.controller.run()

        logger.info(f"Virtual Machine (pid: {self.pid}) launched.")

    @contextlib.contextmanager
    def running_scope(self, timeit: bool = False):
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
            print("Traceback: ", traceback.format_exc())
        else:
            if timeit:
                ed = time.perf_counter_ns()
                print(f"[Timeit] E2E Program Execution Time: {(ed - st) / 1e9} (s).")

    def run(self, coroutine: Coroutine, timeit: bool = False):
        """vm.run method will create a new event loop and run the coroutine."""

        logger.info(f"VM (pid: {self.pid}) runs program: {coroutine.__name__}")

        with self.running_scope(timeit):
            # asyncio.run(coroutine)
            loop = asyncio.new_event_loop()
            loop.run_until_complete(coroutine)
            loop.close()
