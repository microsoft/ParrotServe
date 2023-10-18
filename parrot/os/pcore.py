import json
from typing import Dict
import time

from parrot.program.function import SemanticCall
from parrot.utils import RecyclePool
from parrot.constants import (
    PROCESS_POOL_SIZE,
    ENGINE_POOL_SIZE,
    OS_LOOP_INTERVAL,
    VM_EXPIRE_TIME,
    ENGINE_EXPIRE_TIME,
)
from parrot.engine.config import EngineConfig
from parrot.utils import get_logger

from .config import OSConfig
from .process.process import Process
from .memory.mem_space import MemorySpace
from .engine import ExecutionEngine, EngineRuntimeInfo
from .thread_dispatcher import ThreadDispatcher
from .tokenizer import Tokenizer


logger = get_logger("PCore")


class PCore:
    """Parrot OS Core. It's the entry of the OS-layer of the parrot runtime system.

    It manages the following components:
    - Multiple processes in the frontend.
    - Multiple engines in the backend.
    - Memory space.
    - Thread dispatcher.
    - Tokenizer.
    """

    def __init__(self, os_config_path: str):
        # ---------- Config ----------
        with open(os_config_path, "r") as f:
            self.os_config: OSConfig = json.load(f)

        if self.os_config.max_proc_num > PROCESS_POOL_SIZE:
            logger.warning(
                f"Config max_proc_num: {self.os_config.max_proc_num} larger than "
                "proc_pool_size: {PROCESS_POOL_SIZE}"
            )
        if self.os_config.max_engines_num > ENGINE_POOL_SIZE:
            logger.warning(
                f"Config max_engines_num: {self.os_config.max_engines_num} larger than "
                "engine_pool_size: {ENGINE_POOL_SIZE}"
            )

        # ---------- Components ----------
        self.processes: Dict[int, Process] = {}  # pid -> process
        self.engines: Dict[int, ExecutionEngine] = {}  # engine_id -> engine
        self.mem_space = MemorySpace()
        self.dispatcher = ThreadDispatcher(self.engines)
        self.tokenizer = Tokenizer()

        # ---------- Id Allocator ----------
        self.pid_pool = RecyclePool(PROCESS_POOL_SIZE)
        self.engine_pool = RecyclePool(ENGINE_POOL_SIZE)

        # ---------- Last Seen Time ----------
        self.proc_last_seen_time: Dict[int, float] = {}  # pid -> last_seen_time
        self.engine_last_seen_time: Dict[int, float] = {}  # engine_id -> last_seen_time

    def check_expired(self):
        cur_time = time.perf_counter_ns()

        # VMs
        for pid, last_seen_time in self.proc_last_seen_time.items():
            if cur_time - last_seen_time > VM_EXPIRE_TIME:
                process = self.processes.pop(pid)
                process.free_process()
                self.proc_last_seen_time.pop(pid)
                self.pid_pool.free(pid)
                logger.info(f"VM (pid={pid}) disconnected.")

        # Engines
        for engine_id, last_seen_time in self.engine_last_seen_time.items():
            if cur_time - last_seen_time > ENGINE_EXPIRE_TIME:
                engine = self.engines.pop(engine_id)
                self.engine_last_seen_time.pop(engine_id)
                self.engine_pool.free(engine_id)
                logger.info(f"Engine {engine.name} (id={engine_id}) disconnected.")

    def os_loop(self):
        """Start the OS loop."""

        while True:
            for process in self.processes.values():
                process.monitor_threads()

            self.check_expired()

            time.sleep(OS_LOOP_INTERVAL)

    # ---------- Public APIs ----------

    def register_vm(self) -> int:
        """Register a new VM as a process in the OS."""
        pid = self.pid_pool.allocate()
        process = Process(
            pid=pid,
            dispatcher=self.dispatcher,
            mem_space=self.mem_space,
            tokenizer=self.tokenizer,
        )
        self.processes[pid] = process
        self.proc_last_seen_time[pid] = time.perf_counter_ns()
        logger.info(f"VM (pid={pid}) registered.")
        return pid

    def register_engine(self, name: str, config: EngineConfig) -> int:
        """Register a new engine in the OS."""
        engine_id = self.engine_pool.allocate()
        engine = ExecutionEngine(
            engine_id=engine_id,
            name=name,
            config=config,
        )
        self.engines[engine_id] = engine
        self.engine_last_seen_time[engine_id] = time.perf_counter_ns()
        logger.info(f"Engine {name} (id={engine_id}) registered.")
        return engine_id

    def vm_heartbeat(self, pid: int) -> Dict:
        """Update the last seen time of a VM, and return required data."""

        assert pid in self.processes, f"Unknown pid: {pid}"
        self.proc_last_seen_time[pid] = time.perf_counter_ns()
        logger.info(f"VM (pid={pid}) heartbeat received.")

        mem_used = self.mem_space.profile_process_memory(pid)
        num_threads = len(self.processes[pid].threads)

        return {
            "mem_used": mem_used,
            "num_threads": num_threads,
        }

    def engine_heartbeat(
        self,
        engine_id: int,
        engine_info: EngineRuntimeInfo,
    ):
        """Update the last seen time of an engine and other engine info."""

        assert engine_id in self.engines, f"Unknown engine_id: {engine_id}"
        engine = self.engines[engine_id]
        self.engine_last_seen_time[engine_id] = time.perf_counter_ns()
        engine.runtime_info = engine_info
        logger.info(f"Engine {engine.name} (id={engine_id}) heartbeat received.")

    def submit_call(self, pid: int, call: SemanticCall):
        """Submit a call from a VM to the OS."""

        assert pid in self.processes, f"Unknown pid: {pid}"
        process = self.processes[pid]
        process.execute_call(call)
        logger.info(f'Function call "{call.func.name}" submitted from VM (pid={pid}"')

    async def placeholder_fetch(self, pid: int, placeholder_id: int):
        """Fetch a placeholder content from OS to VM."""

        assert pid in self.processes, f"Unknown pid: {pid}"
        process = self.processes[pid]
        assert (
            placeholder_id in process.placeholders_map
        ), f"Unknown placeholder_id: {placeholder_id}"
        placeholder = process.placeholders_map[placeholder_id]
        logger.info(f'Placeholder (id={placeholder_id}) fetched from VM (pid={pid}"')
        return await placeholder.get()
