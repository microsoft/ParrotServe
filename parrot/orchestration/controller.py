from typing import Dict, List, Union
import logging
import time
import threading
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .engine import ExecutionEngine
from .context import Context
from .tokenize import global_tokenized_storage
from ..utils import get_logger
from ..program.function import ParrotFunction
from ..protocol import check_heartbeat, prefix_init


logger = get_logger("Controller", logging.INFO)


class Controller:
    """Global controller."""

    def __init__(self):
        # ---------- Registry ----------
        self.engines_table: Dict[str, ExecutionEngine] = {}
        self.funtions_table: Dict[str, ParrotFunction] = {}
        self.tokenizers_table: Dict[
            str, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        ] = {}
        self.function_prefix: Dict[str, Context] = {}

        # ---------- Heart Beat ----------
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_daemon, daemon=True
        )
        self._heartbeat_thread.start()

        logger.info("Global controller started.")

    def register_engine(self, name: str, host: str, port: int, tokenizer: str):
        if name in self.engines_table[name]:
            logger.error(f"Engine name {name} has been used.")
            return

        if tokenizer not in self.tokenizers_table:
            logger.error(f"Tokenizer {tokenizer} has not been registered.")
            return

        engine = ExecutionEngine(
            name=name,
            host=host,
            port=port,
            tokenizer=tokenizer,
        )

        try:
            resp = check_heartbeat(engine.http_address)
            assert resp.model_ready
        except:
            return

        engine.cached_tokens = resp.cached_tokens
        engine.running_jobs = resp.running_jobs
        self.engines_table[name] = engine

        logger.info(
            f"Register execution engine: {engine.name} in {engine.http_address}"
        )

    def register_function(
        self,
        function: ParrotFunction,
        caching_prefix: bool = True,
    ):
        if function.name in self.functions_table:
            logger.error(f"Function name {function.name} has been used.")
            return

        if caching_prefix:
            prefix_context = Context()
            for engine in self.engines_table.values():
                prefix_tokens = global_tokenized_storage.tokenize_func_body(
                    function, engine.tokenizer
                )
                try:
                    resp = prefix_init(
                        engine.http_address,
                        prefix_context.context_id,
                        prefix_tokens,
                    )
                    assert resp.filled_tokens_num == len(
                        prefix_tokens
                    ), "Prefix init failed: not all tokens are filled."
                except:
                    return

            self.function_prefix[function.name] = prefix_context

        self.funtions_table[function.name] = function
        logger.info(f"Register parrot function: {function.name}")

    def register_tokenizer(self, tokenizer_name: str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizers_table[tokenizer_name] = tokenizer
        logger.info(f"Register tokenizer: {tokenizer_name}")

    def _heartbeat_daemon(self):
        heartbeat_interval = 5  # (Unit: second)

        while True:
            disconnect_engines: List[str] = []
            for engine in self.engines_table.values():
                try:
                    resp = check_heartbeat(engine.name, engine.http_address)
                except:
                    disconnect_engines.append(engine.name)
                else:
                    # Update engine data
                    engine.cached_tokens = resp.cached_tokens
                    engine.running_jobs = resp.running_jobs

            for engine_name in disconnect_engines:
                self.engines_table.pop(engine_name)

            time.sleep(heartbeat_interval)


# Singleton
parrot_global_ctrl = Controller()
