# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from typing import Optional, Dict, Any
from types import FunctionType

from parrot.utils import get_logger, create_task_in_loop
from parrot.exceptions import parrot_assert
from parrot.serve.graph import NativeFuncNode, ComputeGraph


logger = get_logger("NativeExecutor")


class OutputProxy:

    def __init__(self) -> None:
        self.content: Optional[str] = None

    def set(self, content: str) -> None:
        self.content = content


class PyNativeExecutor:
    """
    PyNativeExecutor in a session directly executes a Python native function in the server side.
    """

    def __init__(self, session_id: int, graph: ComputeGraph) -> None:
        # ---------- Basic Info ----------
        self.session_id = session_id
        self.graph = graph

        # ---------- Execution ----------
        self.func_cache: Dict[str, FunctionType] = {}

        # ---------- Runtime ----------
        self.bad_exception: Optional[Exception] = None

    async def _execute_coroutine(self, func_node: NativeFuncNode) -> None:
        """Coroutine for executing a PyNativeCallRequest."""

        # Block until it's activated by at least 1 SV.
        await func_node.wait_activated()

        # Block until all inputs are ready.
        await func_node.wait_ready()
        native_request = func_node.native_func

        try:
            await asyncio.wait_for(
                self.execute(func_node),
                native_request.metadata.timeout,
            )
        except Exception as e:
            logger.error(
                f"Error when executing Python native function. (func_name={func_node.native_func.func_name}, session_id={self.session_id}): {e}"
            )
            self.exception_interrupt(e)

    def exception_interrupt(self, exception: BaseException):
        self.bad_exception = exception

    def add_native_func(self, func_node: NativeFuncNode) -> None:
        """Add a native function to the graph and assign a coroutine to the request."""

        logger.debug(
            f"Add NativeFunc(request_id={func_node.native_func.request_id}) to executor of Session(session_id={self.session_id})."
        )

        # Insert the request chain into the graph.
        self.graph.insert_native_func_node(func_node)

        # Create execution coroutines for the request chain.
        create_task_in_loop(self._execute_coroutine(func_node))

    async def execute(self, func_node: NativeFuncNode) -> None:
        """Execute a NativeFunc."""

        kwargs: Dict[str, Any] = {}

        # Preparing parameters
        for key, var in func_node.input_vars.items():
            parrot_assert(var.is_ready(), f"Input var {var} is not ready.")
            content = var.get()
            kwargs[key] = content

        for key, value in func_node.input_values.items():
            kwargs[key] = value

        # Create output proxies
        pystring_proxies: Dict[str, OutputProxy] = {}
        for key, var in func_node.output_vars.items():
            pystring_proxies[key] = OutputProxy()
            kwargs[key] = pystring_proxies[key]

        # Execute the native function
        if func_node.native_func.executable_func is not None:
            func_node.native_func.executable_func(**kwargs)
            self.func_cache[func_node.native_func.func_name] = (
                func_node.native_func.executable_func
            )
        else:
            func = self.func_cache.get(func_node.native_func.func_name)
            if func is None:
                raise ValueError(
                    f"Function {func_node.native_func.func_name} not found in Cache."
                )
            func(**kwargs)

        # Write back the output
        for key, var in func_node.output_vars.items():
            var.set(pystring_proxies[key].content)
