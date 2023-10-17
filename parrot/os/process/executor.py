from typing import Dict

from parrot.os.process.tokenizer import Tokenizer
from parrot.utils import get_logger, create_task_in_loop

from .thread import Thread
from .interpreter import TokenIdInterpreter, TextInterpreter, InterpretType


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
        # For TokenIdInterpreter, it is: Tokenizer name -> NativeExecutor
        self.token_id_interpreters: Dict[str, TokenIdInterpreter] = {}
        self.text_interpreter = TextInterpreter()

    def get_token_id_interpreter(self, tokenizer_name: str):
        self.token_id_interpreters[tokenizer_name] = TokenIdInterpreter(
            tokenizer_name,
            self.tokenizer,
        )

    def submit(self, thread: Thread):
        assert thread.dispatched, "Thread must be dispatched before submitting."

        interpret_type = thread.engine.interpreter_type

        if interpret_type == InterpretType.TOKEN_ID:
            interpreter = self.get_token_id_interpreter(
                thread.engine.config.tokenizer_name
            )
        elif interpret_type == InterpretType.TEXT:
            interpreter = self.text_interpreter
        else:
            raise ValueError(f"Unknown interpret type {interpret_type}.")

        interpreter.interpret(thread)

        # Start the thread
        create_task_in_loop(thread.executing())

        logger.info(f"Thread {thread.tid} started.")
