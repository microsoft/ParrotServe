# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import Dict

from parrot.os.tokenizer import Tokenizer
from parrot.utils import get_logger, create_task_in_loop
from parrot.exceptions import ParrotOSInternalError

from .thread import Thread
from .interpreter import TokenIdInterpreter, TextInterpreter
from .interpret_type import InterpretType


logger = get_logger("Executor")


class Executor:
    """Executor is responsible for managing threads and scheduling to execute them.

    A thread submitted to the executor must be dispatched to an engine.
    It will be interpreted and executed by the executor.
    """

    def __init__(self, tokenizer: Tokenizer):
        # ---------- Global components ----------
        self.tokenizer = tokenizer

        # ---------- Sub-executors ----------
        # For TokenIdInterpreter, it is: Tokenizer name -> TokenIdInterpreter
        self.token_id_interpreters: Dict[str, TokenIdInterpreter] = {}
        self.text_interpreter = TextInterpreter()

    def get_token_id_interpreter(self, tokenizer_name: str) -> TokenIdInterpreter:
        if tokenizer_name not in self.token_id_interpreters:
            self.token_id_interpreters[tokenizer_name] = TokenIdInterpreter(
                tokenizer_name,
                self.tokenizer,
            )
        return self.token_id_interpreters[tokenizer_name]

    def submit(self, thread: Thread):
        if not thread.dispatched:
            raise ParrotOSInternalError(
                RuntimeError("Thread must be dispatched before submitting.")
            )

        interpret_type = thread.engine.interpreter_type

        if interpret_type == InterpretType.TOKEN_ID:
            interpreter = self.get_token_id_interpreter(thread.engine.config.tokenizer)
        elif interpret_type == InterpretType.TEXT:
            interpreter = self.text_interpreter
        else:
            raise ParrotOSInternalError(
                ValueError(f"Unknown interpret type {interpret_type}.")
            )

        interpreter.interpret(thread)

        # for operator in thread.operators.queue:
        #     print(operator)

        # Start the thread
        create_task_in_loop(thread.executing())

        logger.info(f"Thread {thread.tid} (pid={thread.process.pid}) started.")
