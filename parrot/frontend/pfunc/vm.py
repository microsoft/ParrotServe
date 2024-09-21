# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import asyncio
import contextlib
import time
import traceback
import importlib
import inspect
from typing import Callable, Optional, Literal, Dict, List, Any, Generator

from parrot.constants import NONE_SESSION_ID

from parrot.protocol.public.apis import (
    register_session,
    get_session_info,
    remove_session,
    submit_semantic_call,
    asubmit_semantic_call,
    register_semantic_variable,
    set_semantic_variable,
    get_semantic_variable,
    aget_semantic_variable,
)

from parrot.utils import time_counter_in_nanoseconds

from .semantic_variable import SemanticVariable
from .perf_criteria import PerformanceCriteria, get_performance_criteria_str
from .function import (
    BasicFunction,
    PyNativeFunction,
    SemanticFunction,
    SemanticCall,
    ParamType,
    Parameter,
)
from parrot.sampling_config import SamplingConfig
from parrot.utils import get_logger


logger = get_logger("PFunc VM")


class VirtualMachine:
    """VirtualMachine for running Parrot semantic programming.

    It represents a session in Parrot's ServeLayer, and proxyes the requests to the session in the ServeLayer.

    It also maintains the function registry.
    """

    def __init__(
        self, core_http_addr: str, mode: Literal["release", "debug"] = "release"
    ) -> None:
        # Public info (User can directly access): core_http_addr, session_id
        self.core_http_addr = core_http_addr

        # Register session and get session_id
        self.session_id = NONE_SESSION_ID
        self._session_auth = ""

        # Function registry
        self._function_registry: Dict[str, BasicFunction] = {}
        self._anonymous_funcname_counter = 0

        self.stat_run_time = 0.0

        if mode == "release":
            import logging

            # We don't disable the error log
            logging.disable(logging.DEBUG)
            logging.disable(logging.INFO)

    def _get_session_id_str(self) -> str:
        return "NONE" if self.session_id == NONE_SESSION_ID else f"{self.session_id}"

    # ----------Methods for Program Interface ----------

    def register_semantic_variable_handler(self, var_name: str) -> str:
        """Register a semantic variable to the VM.

        Args:
            var_name: str. The name of the variable.

        Returns:
            str: The id of the variable.
        """

        resp = register_semantic_variable(
            http_addr=self.core_http_addr,
            session_id=self.session_id,
            session_auth=self._session_auth,
            var_name=var_name,
        )

        var_id = resp.var_id

        logger.info(
            f"VM (session_id={self._get_session_id_str()}) registers SemanticVariable: {var_name} (id={var_id})"
        )

        return var_id

    def set_semantic_variable_handler(self, var_id: str, content: str) -> None:
        """Set the content of a SemanticVariable.

        Args:
            var_id: str. The id of the SemanticVariable.
            content: str. The content to be set.
        """

        resp = set_semantic_variable(
            http_addr=self.core_http_addr,
            session_id=self.session_id,
            session_auth=self._session_auth,
            var_id=var_id,
            content=content,
        )

    def get_semantic_variable_handler(
        self, var_id: str, criteria: PerformanceCriteria
    ) -> str:
        """Fetch the content of a SemanticVariable.

        Args:
            var_id: str. The id of the SemanticVariable.
            criteria: PerformanceCriteria. The performance criteria for fetching the variable.

        Returns:
            str: The content of the SemanticVariable.
        """

        resp = get_semantic_variable(
            http_addr=self.core_http_addr,
            session_id=self.session_id,
            session_auth=self._session_auth,
            var_id=var_id,
            criteria=get_performance_criteria_str(criteria),
        )
        return resp.content

    async def aget_semantic_variable_handler(
        self, var_id: str, criteria: PerformanceCriteria
    ) -> str:
        """(Async) Fetch the content of a SemanticVariable.

        Args:
            var_id: str. The id of the SemanticVariable.
            criteria: PerformanceCriteria. The performance criteria for fetching the variable.

        Returns:
            str: The content of the SemanticVariable.
        """

        resp = await aget_semantic_variable(
            http_addr=self.core_http_addr,
            session_id=self.session_id,
            session_auth=self._session_auth,
            var_id=var_id,
            criteria=get_performance_criteria_str(criteria),
        )
        return resp.content

    def register_function_handler(self, func: BasicFunction) -> None:
        """Register a function to the VM."""

        if func.name in self._function_registry:
            # raise ValueError(f"Function {func.name} already registered.")
            # Don't raise error here, because we may register the same function
            return

        self._function_registry[func.name] = func
        logger.info(
            f"VM (session_id={self._get_session_id_str()}) registers function: {func.name}"
        )

    def submit_semantic_call_handler(self, call: SemanticCall) -> List:
        """Submit a SemanticCall to the ServeCore.

        Args:
            call: SemanticCall. The call to be submitted.

        Returns:
            Dict. The "param info" returned by the ServeCore.
        """

        logger.info(
            f"VM (session_id={self._get_session_id_str()}) submits SemanticCall: {call.func.name}"
        )

        resp = submit_semantic_call(
            http_addr=self.core_http_addr,
            session_id=self.session_id,
            session_auth=self._session_auth,
            payload=call.to_request_payload(),
        )

        return resp.param_info

    async def asubmit_semantic_call_handler(self, call: SemanticCall) -> List:
        """Submit a call to the ServeCore.

        Args:
            call: SemanticCall. The call to be submitted.

        Returns:
            Dict. The "param info" returned by the ServeCore.
        """

        logger.info(
            f"VM (session_id={self._get_session_id_str()}) submits SemanticCall: {call.func.name}"
        )

        resp = await asubmit_semantic_call(
            http_addr=self.core_http_addr,
            session_id=self.session_id,
            session_auth=self._session_auth,
            payload=call.to_request_payload(),
        )

        return resp.param_info

    # ---------- Public Methods ----------

    @property
    def session_registered(self) -> bool:
        return self.session_id != NONE_SESSION_ID

    def register_session(self) -> None:
        """Register a session to the ServeCore."""

        resp = register_session(http_addr=self.core_http_addr, api_key="1")
        self.session_id = resp.session_id
        self._session_auth = resp.session_auth

        logger.info(
            f"VM registered a Session (session_id={self._get_session_id_str()})."
        )

    def unregister_session(self) -> None:
        """Unregister the session from the ServeCore."""

        if not self.session_registered:
            return

        remove_session(
            http_addr=self.core_http_addr,
            session_id=self.session_id,
            session_auth=self._session_auth,
        )

        logger.info(
            f"VM unregistered its Session (session_id={self._get_session_id_str()})."
        )
        self.session_id = NONE_SESSION_ID
        self._session_auth = ""

    def define_function(
        self,
        func_name: Optional[str],
        func_body: str,
        params: List[Parameter],
        try_register: bool = True,
        **semantic_func_metadata,
    ) -> SemanticFunction:
        if func_name is None:
            func_name = f"anonymous_{self._anonymous_funcname_counter}"
            self._anonymous_funcname_counter += 1

        for param in params:
            if param.typ == ParamType.OUTPUT_LOC:
                if param.sampling_config is None:
                    param.sampling_config = SamplingConfig()

        func = SemanticFunction(
            name=func_name,
            func_body_str=func_body,
            params=params,
            try_register=try_register,
            **semantic_func_metadata,
        )

        self.register_function_handler(func)

        return func

    def import_function(
        self,
        function_name: str,
        module_path: str,
    ) -> SemanticFunction:
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

    def set_global_env(self) -> None:
        """Set the global environment for current Python process."""

        BasicFunction._virtual_machine_env = self
        SemanticVariable._virtual_machine_env = self
        # SharedContext._controller = self.controller
        # SharedContext._tokenized_storage = self.tokenizer

        self.register_session()

    def unset_global_env(self) -> None:
        """Unset the global environment for current Python process."""

        BasicFunction._virtual_machine_env = None
        SemanticVariable._virtual_machine_env = self
        # SharedContext._controller = None
        # SharedContext._tokenized_storage = None

        self.unregister_session()

    @contextlib.contextmanager
    def running_scope(self, timeit: bool = False) -> Generator[Any, Any, Any]:
        """Any code that runs under this scope will be executed under the VM context.

        - For native code, it will be executed by the system Python interpreter.
        - For semantic code, it will be submitted to the OS and executed finally
          by Parrot backend engines.
        """

        self.set_global_env()

        if timeit:
            st = time_counter_in_nanoseconds()

        try:
            yield
        except BaseException as e:
            # NOTE(chaofan): This is mainly used to catch the error in the `main`.
            #
            # For errors in programs, we use the fail fast mode and quit the whole system
            # In this case, we can only see a SystemExit error
            print("Error happens when executing Parrot program: ", type(e), repr(e))
            print("Traceback: ", traceback.format_exc())
            self.unset_global_env()
        else:
            self.unset_global_env()
            if timeit:
                ed = time_counter_in_nanoseconds()
                self.stat_run_time = (ed - st) / 1e9  # Convert to seconds
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

        logger.info(
            f"VM (session_id={self._get_session_id_str()}) runs program: {program.__name__}"
        )

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
