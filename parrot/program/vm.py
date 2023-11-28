# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import asyncio
import contextlib
import time
import traceback
import threading
import importlib
from dataclasses import dataclass
from typing import Callable, Coroutine, Literal, Dict, List, Any


from parrot.protocol.layer_apis import (
    register_vm,
    vm_heartbeat,
    submit_call,
    asubmit_call,
    placeholder_set,
    placeholder_fetch,
    aplaceholder_fetch,
)
from parrot.program.function import SemanticFunction, Future, SemanticCall
from parrot.utils import get_logger
from parrot.constants import VM_HEARTBEAT_INTERVAL, NONE_CONTEXT_ID


logger = get_logger("VM")


@dataclass
class VMRuntimeInfo:
    mem_used: float = 0
    num_threads: int = 0


class VirtualMachine:
    """The Virtual Machine for Parrot semantic programming.

    Different from the traditional VM, there is no complex execution logic in Parrot VM.
    Instead, it's more like a client, which sends semantc function calls to OS and waits for
    the results.
    """

    def __init__(
        self, os_http_addr: str, mode: Literal["release", "debug"] = "release"
    ):
        # Public info (User can directly access): os_http_addr, pid, runtime_info

        self.os_http_addr = os_http_addr

        # Call OS to register VM, and allocate a pid
        resp = register_vm(self.os_http_addr)

        self.pid = resp.pid
        self.runtime_info = VMRuntimeInfo()

        self.stat_run_time = 0.0

        if mode == "release":
            import logging

            # We don't disable the error log
            logging.disable(logging.DEBUG)
            logging.disable(logging.INFO)

        # The following attributes are internal.

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_daemon, daemon=True
        )
        self._function_registry: Dict[str, SemanticFunction] = {}
        self._heartbeat_thread.start()

        logger.info(f"Virtual Machine (pid: {self.pid}) launched.")

    # ---------- Private Methods ----------

    def _heartbeat_daemon(self):
        while True:
            resp = vm_heartbeat(
                http_addr=self.os_http_addr,
                pid=self.pid,
            )

            self.runtime_info = VMRuntimeInfo(**resp.dict())

            time.sleep(VM_HEARTBEAT_INTERVAL)

    # ----------Methods for Program Interface ----------

    def placeholder_set_handler(self, placeholder_id: int, content: str):
        """Set the content of a placeholder to OS."""

        resp = placeholder_set(
            http_addr=self.os_http_addr,
            pid=self.pid,
            placeholder_id=placeholder_id,
            content=content,
        )

    def placeholder_fetch_handler(self, placeholder_id: int) -> str:
        """Fetch a placeholder from OS."""

        resp = placeholder_fetch(
            http_addr=self.os_http_addr,
            pid=self.pid,
            placeholder_id=placeholder_id,
        )
        return resp.content

    async def aplaceholder_fetch_handler(self, placeholder_id: int) -> str:
        """(Async) fetch a placeholder from OS."""

        resp = await aplaceholder_fetch(
            http_addr=self.os_http_addr,
            pid=self.pid,
            placeholder_id=placeholder_id,
        )
        return resp.content

    def register_function_handler(self, func: SemanticFunction):
        """Register a semantic function to the VM."""

        if func.name in self._function_registry:
            raise ValueError(f"Function {func.name} already registered.")

        self._function_registry[func.name] = func
        logger.info(f"VM (pid: {self.pid}) registers function: {func.name}")

    def submit_call_handler(self, call: SemanticCall):
        """Submit a call to the OS."""

        logger.info(f"VM (pid: {self.pid}) submits call: {call.func.name}")

        resp = submit_call(
            http_addr=self.os_http_addr,
            pid=self.pid,
            call=call,
        )

    async def asubmit_call_handler(self, call: SemanticCall):
        """Submit a call to the OS."""

        logger.info(f"VM (pid: {self.pid}) submits call: {call.func.name}")

        resp = await asubmit_call(
            http_addr=self.os_http_addr,
            pid=self.pid,
            call=call,
        )

    # ---------- Public Methods ----------

    def import_function(
        self,
        function_name: str,
        module_path: str,
    ):
        """Import a semantic function from a Python module.

        - The function name is the name of the semantic function in the module file;
        - The module path is in the format of `xx.yy.zz`, with the root directory
          being the Parrot root directory.
        """

        try:
            module = importlib.import_module(f"{module_path}")
            semantic_function = getattr(module, function_name)
        except:
            raise ImportError(
                f"Cannot import function {function_name} from module: {module_path}."
            )

        if not isinstance(semantic_function, SemanticFunction):
            raise ValueError(f"Function {function_name} is not a semantic function.")

        self.register_function_handler(semantic_function)
        return semantic_function

    def set_global_env(self):
        """Set the global environment for current Python process."""

        SemanticFunction._virtual_machine_env = self
        Future._virtual_machine_env = self
        # SharedContext._controller = self.controller
        # SharedContext._tokenized_storage = self.tokenizer

    def unset_global_env(self):
        """Unset the global environment for current Python process."""

        SemanticFunction._virtual_machine_env = None
        Future._virtual_machine_env = None
        # SharedContext._controller = None
        # SharedContext._tokenized_storage = None

    @contextlib.contextmanager
    def running_scope(self, timeit: bool = False):
        """Any code that runs under this scope will be executed under the VM context.

        - For native code, it will be executed by the system Python interpreter.
        - For semantic code, it will be submitted to the OS and executed finally
          by Parrot backend engines.
        """

        self.set_global_env()

        if timeit:
            st = time.perf_counter_ns()

        try:
            yield
        except BaseException as e:
            # NOTE(chaofan): This is mainly used to catch the error in the `main`.
            #
            # For errors in programs, we use the fail fast mode and quit the whole system
            # In this case, we can only see a SystemExit error
            print("Error happens when executing Parrot program: ", type(e), repr(e))
            print("Traceback: ", traceback.format_exc())
        else:
            self.unset_global_env()
            if timeit:
                ed = time.perf_counter_ns()
                self.stat_run_time = (ed - st) / 1e9
                print(f"[Timeit] E2E Program Execution Time: {self.stat_run_time} (s).")

    def run(
        self,
        program: Callable,
        timeit: bool = False,
        args: List[Any] = [],
    ) -> float:
        """vm.run method wraps a E2E running process of a semantic program.

        It accepts both normal functions and coroutines. When the program is a coroutine,
        VM will create a new event loop and run the coroutine.

        Args are the arguments passed to the program. This case is only used for normal functions.

        Return the E2E running time of the program.
        """

        logger.info(f"VM (pid: {self.pid}) runs program: {program.__name__}")

        with self.running_scope(timeit):
            # asyncio.run(program)
            if isinstance(program, Coroutine):
                loop = asyncio.new_event_loop()
                loop.run_until_complete(program)
                loop.close()
            else:
                program(*args)

        return self.stat_run_time

    def profile(
        self,
        program: Callable,
        warmups: int = 3,
        trials: int = 20,
        args: List[Any] = [],
    ) -> float:
        """Profile the E2E lantecy of certain semantic program."""

        sleep_interval = 2.5

        for _ in range(warmups):
            self.run(program, args)
            time.sleep(sleep_interval)

        e2e_lantecy = 0.0

        for _ in range(trials):
            self.run(program, timeit=True, args=args)
            e2e_lantecy += self.stat_run_time
            time.sleep(sleep_interval)

        return e2e_lantecy / trials
