# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import asyncio
import contextlib
import time
import traceback
import threading
import importlib
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Literal, Dict, List, Any


from parrot.exceptions import parrot_assert
from parrot.protocol.runtime_info import VMRuntimeInfo
from parrot.protocol.layer_apis import (
    register_vm,
    vm_heartbeat,
    submit_call,
    asubmit_call,
    placeholder_set,
    placeholder_fetch,
    aplaceholder_fetch,
)
from parrot.program.semantic_variable import SemanticVariable
from parrot.program.function import (
    BasicFunction,
    NativeFunction,
    SemanticFunction,
    ParamType,
    Parameter,
)
from parrot.protocol.annotation import DispatchAnnotation
from parrot.protocol.sampling_config import SamplingConfig
from parrot.program.function_call import BasicCall
from parrot.utils import get_logger
from parrot.constants import VM_HEARTBEAT_INTERVAL, NONE_CONTEXT_ID


logger = get_logger("VM")


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
        self._function_registry: Dict[str, BasicFunction] = {}
        self._heartbeat_thread.start()

        self._anonymous_funcname_counter = 0

        self._batch = []
        self._batch_latency = 0.0
        self._batch_flag = False

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

    def _batch_hook(self, call: BasicCall) -> bool:
        if not self._batch_flag:
            return False

        is_native = isinstance(call.func, NativeFunction)

        async def _submit_call(idx: int):
            if self._batch_latency > 0:
                await asyncio.sleep(idx * self._batch_latency)  # Order the batch, 1ms latency
            await asubmit_call(
                http_addr=self.os_http_addr,
                pid=self.pid,
                call=call,
                is_native=is_native,
            )

        self._batch.append(_submit_call(len(self._batch)))

        return True

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

    def register_function_handler(self, func: BasicFunction):
        """Register a function to the VM."""

        if func.name in self._function_registry:
            # raise ValueError(f"Function {func.name} already registered.")
            # Don't raise error here, because we may register the same function
            return

        self._function_registry[func.name] = func
        logger.info(f"VM (pid: {self.pid}) registers function: {func.name}")

    def submit_call_handler(self, call: BasicCall):
        """Submit a call to the OS."""

        # If batching
        if self._batch_hook(call):
            return

        logger.info(f"VM (pid: {self.pid}) submits call: {call.func.name}")

        is_native = isinstance(call.func, NativeFunction)

        resp = submit_call(
            http_addr=self.os_http_addr,
            pid=self.pid,
            call=call,
            is_native=is_native,
        )

    async def asubmit_call_handler(self, call: BasicCall):
        """Submit a call to the OS."""

        logger.info(f"VM (pid: {self.pid}) submits call: {call.func.name}")

        is_native = isinstance(call.func, NativeFunction)

        resp = await asubmit_call(
            http_addr=self.os_http_addr,
            pid=self.pid,
            call=call,
            is_native=is_native,
        )

    # ---------- Public Methods ----------

    # Submit batching calls with ordering!

    def set_batch(self, latency: float = 0.001):
        """Set the batch flag to True."""

        parrot_assert(not self._batch_flag, "Batching is already set.")
        self._batch_flag = True
        self._batch_latency = latency

    async def submit_batch(self):
        """Submit the batch to OS."""

        parrot_assert(self._batch_flag, "Batching is not set.")
        self._batch_flag = False
        self._batch_latency = 0.001

        await asyncio.gather(*self._batch)
        self._batch = []

    def define_function(
        self,
        func_name: Optional[str],
        func_body: str,
        params: List[Parameter],
        models: List[str] = [],
        cache_prefix: bool = True,
        remove_pure_fill: bool = True,
    ) -> SemanticFunction:
        if func_name is None:
            func_name = f"anonymous_{self._anonymous_funcname_counter}"
            self._anonymous_funcname_counter += 1

        for param in params:
            if param.typ == ParamType.OUTPUT_LOC:
                if param.dispatch_annotation is None:
                    param.dispatch_annotation = DispatchAnnotation()
                if param.sampling_config is None:
                    param.sampling_config = SamplingConfig()

        func = SemanticFunction(
            name=func_name,
            func_body_str=func_body,
            params=params,
            # Func Metadata
            models=models,
            cache_prefix=cache_prefix,
            remove_pure_fill=remove_pure_fill,
        )

        self.register_function_handler(func)

        return func

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

        if not isinstance(semantic_function, BasicFunction):
            raise ValueError(
                f"Function {function_name} is not a semantic function or a native function."
            )

        self.register_function_handler(semantic_function)
        return semantic_function

    def set_global_env(self):
        """Set the global environment for current Python process."""

        BasicFunction._virtual_machine_env = self
        SemanticVariable._virtual_machine_env = self
        # SharedContext._controller = self.controller
        # SharedContext._tokenized_storage = self.tokenizer

    def unset_global_env(self):
        """Unset the global environment for current Python process."""

        BasicFunction._virtual_machine_env = None
        SemanticVariable._virtual_machine_env = self
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
                logger.info(
                    f"[Timeit] E2E Program Execution Time: {self.stat_run_time} (s)."
                )

    def run(
        self,
        program: Callable,
        timeit: bool = False,
        args: List[Any] = [],
    ) -> float:
        """vm.run method wraps a E2E running process of a semantic program.

        It accepts both normal functions and async functions. When the program is async,
        VM will create a new event loop and run the coroutine it created.

        For simplicity, we only support positional arguments for now.

        Return the E2E running time of the program.
        """

        logger.info(f"VM (pid: {self.pid}) runs program: {program.__name__}")

        if inspect.iscoroutinefunction(program):
            coroutine = program(*args)
        else:
            coroutine = None

        with self.running_scope(timeit):
            # asyncio.run(program)
            # if isinstance(program, Coroutine):

            if coroutine:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(coroutine)
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
