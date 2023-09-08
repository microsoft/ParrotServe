from typing import Dict, List, Union, Optional
import logging
import time
import threading
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .engine import ExecutionEngine
from .context import Context
from ..utils import get_logger
from ..program.function import ParrotFunction
from ..protocol import check_heartbeat, prefix_init


logger = get_logger("Controller", logging.INFO)


class Controller:
    """Global controller."""

    def __init__(self):
        # ---------- Registry ----------
        self.engines_table: Dict[str, ExecutionEngine] = {}
        self.functions_table: Dict[str, ParrotFunction] = {}
        self.tokenizers_table: Dict[
            str, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        ] = {}
        self.function_prefix: Dict[str, Context] = {}

        # ---------- Control components ----------
        self.executor: Optional["Executor"] = None

        # ---------- Flag ----------
        self._run_flag = False

        # ---------- Heart Beat ----------
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_daemon, daemon=True
        )

    def _check_is_run(self):
        if self._run_flag:
            raise RuntimeError("Controller is running now, can't register/rerun.")

    @property
    def is_running(self):
        return self._run_flag

    def run(self):
        self._check_is_run()
        self._run_flag = True
        self._heartbeat_thread.start()
        logger.info("Global controller started.")

    def caching_function_prefix(self, tokenized_storage: "TokenizedStorage"):
        for func_name, prefix_context in self.function_prefix.items():
            func = self.functions_table[func_name]
            for engine in self.engines_table.values():
                prefix_tokens = tokenized_storage.tokenize_func_body(
                    func, engine.tokenizer
                )[0]
                resp = prefix_init(
                    engine.http_address,
                    prefix_context.context_id,
                    prefix_tokens,
                )
                assert resp.filled_tokens_num == len(
                    prefix_tokens
                ), "Prefix init failed: not all tokens are filled."

    def register_engine(self, name: str, host: str, port: int, tokenizer: str):
        self._check_is_run()

        if name in self.engines_table:
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
            resp = check_heartbeat(engine.name, engine.http_address)
        except:
            logger.error(f"Register engine {engine.name} failed.")
            return
        assert resp.model_ready, "Engine is not ready."

        engine.cached_tokens = resp.cached_tokens
        engine.running_jobs = resp.running_jobs
        self.engines_table[name] = engine

        logger.info(
            f"Register execution engine: {engine.name} in {engine.http_address}"
        )

    def register_function(self, function: ParrotFunction, caching_prefix: bool):
        self._check_is_run()

        if function.name in self.functions_table:
            logger.error(f"Function name {function.name} has been used.")
            return

        if caching_prefix:
            prefix_context = Context()
            self.function_prefix[function.name] = prefix_context

        self.functions_table[function.name] = function
        logger.info(f"Register parrot function: {function.name}")

    def register_tokenizer(self, tokenizer_name: str):
        self._check_is_run()

        if tokenizer_name in self.tokenizers_table:
            logger.error(f"Tokenizer name {tokenizer_name} has been used.")
            return

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizers_table[tokenizer_name] = tokenizer

        # executor=None: testing mode
        if self.executor is not None:
            self.executor.register_group_executor(tokenizer_name)
        else:
            logger.warning(
                "Executor is not initialized. Not register the group executor."
            )

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
                logger.info(f"Engine {engine_name} disconnected.")

            time.sleep(heartbeat_interval)
