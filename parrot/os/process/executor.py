from typing import Dict

from parrot.program.function import SemanticCall
from Parrot.parrot.os.process.tokenizer import Tokenizer
from parrot.protocol.thread_metadata import ThreadMetadata
from parrot.protocol.layer_apis import thread_start
from parrot.utils import get_logger

from .thread import Thread
from .interpreter import TokenIdInterpreter, TextInterpreter, InterpretType


logger = get_logger("Executor")


class Executor:
    """Executor is responsible for managing calls and scheduling to execute them."""

    def __init__(self, vm: "VirtualMachine", tokenizer: Tokenizer):
        # ---------- Global components ----------
        self.vm = vm
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

    def submit(self, call: SemanticCall):
        thread = Thread(vm=self.vm, call=call)

        metadata = ThreadMetadata(is_latency_critical=True)

        # Call thread dispatcher
        resp = thread_start(
            http_addr=self.vm.os_http_addr,
            pid=self.vm.pid,
            tid=thread.tid,
            metadata=metadata,
        )

        if resp.interpret_type == InterpretType.TOKEN_ID:
            interpreter = self.get_token_id_interpreter(resp.tokenizer)
        elif resp.interpret_type == InterpretType.TEXT:
            interpreter = self.text_interpreter
        else:
            raise ValueError(f"Unknown interpret type {resp.interpret_type}.")

        interpreter.interpret(thread)

        logger.info(f"SemanticCall {call.func.name} created a thread {thread.tid}.")
