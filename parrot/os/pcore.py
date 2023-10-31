import json
from typing import Dict, List
import asyncio
import time
from dataclasses import asdict

from parrot.program.vm import VMRuntimeInfo
from parrot.program.function import SemanticCall
from parrot.utils import RecyclePool
from parrot.constants import (
    PROCESS_POOL_SIZE,
    ENGINE_POOL_SIZE,
    OS_LOOP_INTERVAL,
    VM_EXPIRE_TIME,
    ENGINE_EXPIRE_TIME,
)
from parrot.protocol.layer_apis import ping_engine
from parrot.engine.config import EngineConfig
from parrot.utils import get_logger
from parrot.exceptions import ParrotOSUserError, ParrotOSInteralError

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
            self.os_config = dict(json.load(f))

        # if not OSConfig.verify_config(self.os_config):
        #     raise ParrotOSInteralError(
        #         ValueError(f"Invalid OS config: {self.os_config}")
        #     )

        self.os_config = OSConfig(**self.os_config)

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

        def flush_engine_callback():
            self._ping_engines()

        self.dispatcher = ThreadDispatcher(
            engines=self.engines, flush_engine_callback=flush_engine_callback
        )
        self.tokenizer = Tokenizer()

        # ---------- Id Allocator ----------
        self.pid_pool = RecyclePool(PROCESS_POOL_SIZE)
        self.engine_pool = RecyclePool(ENGINE_POOL_SIZE)

        # ---------- Last Seen Time ----------
        self.proc_last_seen_time: Dict[int, float] = {}  # pid -> last_seen_time
        self.engine_last_seen_time: Dict[int, float] = {}  # engine_id -> last_seen_time

        logger.info(
            f"PCore started with config: \n"
            + "\n".join(
                [f"  {key}={value}, " for key, value in self.os_config.__dict__.items()]
            )
        )

    def _check_expired(self):
        cur_time = time.perf_counter_ns()

        # VMs
        for pid, last_seen_time in self.proc_last_seen_time.items():
            if (cur_time - last_seen_time) / 1e9 > VM_EXPIRE_TIME:
                self.processes[pid].dead = True

        # Engines
        for engine_id, last_seen_time in self.engine_last_seen_time.items():
            if (cur_time - last_seen_time) / 1e9 > ENGINE_EXPIRE_TIME:
                self.engines[engine_id].dead = True

    def _ping_engines(self):
        for engine in self.engines.values():
            if not engine.dead:
                pong = ping_engine(engine.http_address)
                if not pong:
                    engine.dead = True

    def _sweep_dead_clients(self):
        dead_procs: List[Process] = [
            proc for proc in self.processes.values() if proc.dead
        ]
        dead_engines: List[ExecutionEngine] = [
            engine for engine in self.engines.values() if engine.dead
        ]

        # VMs
        for process in dead_procs:
            pid = process.pid
            self.processes.pop(pid)
            # If a VM is dead, we need to free all its resources (garbage collection).
            process.free_process()
            self.proc_last_seen_time.pop(pid)
            self.pid_pool.free(pid)
            logger.info(f"VM (pid={pid}) disconnected.")

        # Engines
        for engine in dead_engines:
            engine_id = engine.engine_id
            self.engines.pop(engine_id)
            self.engine_last_seen_time.pop(engine_id)
            self.engine_pool.free(engine_id)
            logger.info(f"Engine {engine.name} (id={engine_id}) disconnected.")

    def _check_process(self, pid: int):
        if pid not in self.processes:
            raise ParrotOSUserError(ValueError(f"Unknown pid: {pid}"))

        process = self.processes[pid]
        if process.dead:
            raise ParrotOSUserError(RuntimeError(f"Process (pid={pid}) is dead."))

        if process.bad:
            process.dead = True
            raise ParrotOSUserError(
                RuntimeError(
                    f"Process (pid={pid}) is bad because exceptions happen "
                    "during the execution of threads."
                )
            )

    # ---------- Public APIs ----------

    async def os_loop(self):
        """Start the OS loop."""

        while True:
            for process in self.processes.values():
                if process.live:
                    process.monitor_threads()

            self._check_expired()
            self._sweep_dead_clients()

            await asyncio.sleep(OS_LOOP_INTERVAL)

    def register_vm(self) -> int:
        """Register a new VM as a process in the OS."""
        pid = self.pid_pool.allocate()
        process = Process(
            pid=pid,
            dispatcher=self.dispatcher,
            memory_space=self.mem_space,
            tokenizer=self.tokenizer,
        )
        self.processes[pid] = process
        self.proc_last_seen_time[pid] = time.perf_counter_ns()
        logger.info(f"VM (pid={pid}) registered.")
        return pid

    def register_engine(self, config: EngineConfig) -> int:
        """Register a new engine in the OS."""
        engine_id = self.engine_pool.allocate()
        engine = ExecutionEngine(
            engine_id=engine_id,
            config=config,
        )
        self.engines[engine_id] = engine
        self.engine_last_seen_time[engine_id] = time.perf_counter_ns()
        logger.info(f"Engine {engine.name} (id={engine_id}) registered.")
        return engine_id

    def vm_heartbeat(self, pid: int) -> Dict:
        """Update the last seen time of a VM, and return required data."""

        self._check_process(pid)

        self.proc_last_seen_time[pid] = time.perf_counter_ns()
        logger.info(f"VM (pid={pid}) heartbeat received.")

        mem_used = self.mem_space.profile_process_memory(pid)
        num_threads = len(self.processes[pid].threads)

        return asdict(
            VMRuntimeInfo(
                mem_used=mem_used,
                num_threads=num_threads,
            )
        )

    def engine_heartbeat(
        self,
        engine_id: int,
        engine_info: EngineRuntimeInfo,
    ):
        """Update the last seen time of an engine and other engine info."""

        if engine_id not in self.engines:
            raise ParrotOSUserError(ValueError(f"Unknown engine_id: {engine_id}"))
        engine = self.engines[engine_id]
        self.engine_last_seen_time[engine_id] = time.perf_counter_ns()
        engine.runtime_info = engine_info
        logger.info(f"Engine {engine.name} (id={engine_id}) heartbeat received.")

    def submit_call(self, pid: int, call: SemanticCall, context_id: int) -> int:
        """Submit a call from a VM to the OS."""

        self._check_process(pid)

        process = self.processes[pid]
        st = time.perf_counter_ns()
        thread = process.execute_call(call, context_id)
        ed = time.perf_counter_ns()

        logger.info(
            f'Function call "{call.func.name}" submitted from VM (pid={pid}). '
            f"Time used: {(ed - st) / 1e9} s."
        )

        return thread.ctx.context_id

    async def placeholder_fetch(self, pid: int, placeholder_id: int):
        """Fetch a placeholder content from OS to VM."""

        self._check_process(pid)

        process = self.processes[pid]
        if placeholder_id not in process.placeholders_map:
            raise ParrotOSUserError(
                ValueError(f"Unknown placeholder_id: {placeholder_id}")
            )
        placeholder = process.placeholders_map[placeholder_id]
        logger.info(f"Placeholder (id={placeholder_id}) fetched from VM (pid={pid})")
        return await placeholder.get()
