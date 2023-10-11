from typing import Dict, List
import logging
import time
import threading

from parrot.utils import get_logger
from parrot.program.function import SemanticFunction
from parrot.constants import HEARTBEAT_INTERVAL
from parrot.protocol.layer_apis import vm_heartbeat


logger = get_logger("Controller", logging.INFO)


class Controller:
    """Global controller in VM.

    It does the following things:
    - Function registration.
    - VM Heartbeat.
    """

    def __init__(self, vm: "VirtualMachine"):
        # ---------- Basic Members ----------
        self.vm = vm
        self._run_flag = False

        # ---------- Function Registration ----------
        self.functions_table: Dict[str, SemanticFunction] = {}

        # ---------- Heartbeat ----------
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_daemon, daemon=True
        )

        # ---------- OS info ----------
        self.available_models_in_os: List[str] = {}

    def _check_is_run(self):
        if self.is_running:
            raise RuntimeError("Controller is running now, can't register/rerun.")

    @property
    def is_running(self):
        return self._run_flag

    def run(self):
        self._check_is_run()
        self._run_flag = True
        self._heartbeat_thread.start()
        logger.info(f"Global controller started. Client UUID: {self.client_id}")

    def register_function(self, function: SemanticFunction):
        self._check_is_run()

        assert (
            function.name not in self.functions_table
        ), f"Function name {function.name} has been used."

        self.functions_table[function.name] = function
        logger.info(f"Register parrot function: {function.name}")

    def _heartbeat_daemon(self):
        while True:
            resp = vm_heartbeat(
                http_addr=self.vm.os_http_address,
                pid=self.vm.pid,
            )

            # Update OS info
            self.available_models_in_os = resp["available_models"]

            time.sleep(HEARTBEAT_INTERVAL)
