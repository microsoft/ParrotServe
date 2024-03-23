# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum
from typing import List, Dict

from parrot.protocol.internal.runtime_info import EngineRuntimeInfo

# WARN(chaofan): Import from engine package
from parrot.engine.config import EngineConfig

from parrot.constants import (
    ENGINE_TYPE_BUILTIN,
    ENGINE_TYPE_OPENAI,
)
from parrot.exceptions import parrot_assert


from ..tokenizer_wrapper import TokenizersWrapper

from .model import LanguageModel, ModelType


class EngineStatus(Enum):
    RUNNING = 0  # The engine is running
    DEAD = 1  # The engine is dead, i.e. the heartbeat is not received
    BAD = 2  # The engine is bad, i.e. an exception is raised from the engine


class ServeLayerRuntimeInfo:
    """Serve-layer runtime info of an engine."""

    def __init__(self):
        self.num_tasks = 0
        self.tokens_num = 0


class ExecutionEngine:
    """Represent an execution engine in the backend."""

    def __init__(
        self,
        engine_id: int,
        config: EngineConfig,
        model: LanguageModel,
    ):
        # ---------- Basic Config ----------
        self.engine_id = engine_id
        self.config = config
        self.model = model

        # ---------- Status ----------
        self.status: EngineStatus = EngineStatus.RUNNING

        # ---------- Runtime Info ----------

        # NOTE(chaofan): There are two info packages in the runtime info:
        # - Real-time info: The info that changes frequently, like the number of running jobs.
        #                   This info type is sent from engine to OS in heartbeat messages.
        # - Serve-layer info: Info maintained by ServeCore.
        #
        # Synchronization between these two info packages is necessary: by updating the static info
        # when the real-time info changes.

        self.real_time_runtime_info = EngineRuntimeInfo()
        self.serve_layer_runtime_info = ServeLayerRuntimeInfo()

    # ---------- Status Methods ----------

    def mark_dead(self) -> None:
        self.status = EngineStatus.DEAD

    def mark_bad(self) -> None:
        self.status = EngineStatus.BAD

    @property
    def not_running(self) -> bool:
        return self.status != EngineStatus.RUNNING

    # ---------- Basic Info ----------

    @property
    def name(self) -> str:
        return self.config.engine_name

    @property
    def http_address(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def model_type(self) -> ModelType:
        return self.model.model_type

    @property
    def requires_token_ids(self) -> bool:
        return self.model_type == ModelType.TOKEN_ID

    # ---------- For Scheduling ----------

    @property
    def remain_thread_locs(self) -> int:
        return self.config.threads_capacity - self.num_threads

    @property
    def num_threads(self) -> int:
        return len(self.threads)

    @property
    def requests_num_upperbound(self) -> int:
        """Return the upperbound of the number of jobs that can be dispatched to this self."""
        return min(
            [self.config.threads_capacity]
            + [thread.requests_num_upperbound for thread in self.threads]
        )

    def count_thread_token_nums(self, thread: "Thread") -> int:
        if self.config.engine_type != ENGINE_TYPE_BUILTIN:
            return 0  # TODO(chaofan): support other engines

        tokenizer_name = self.config.tokenizer

        length = 0

        # Count the length of the thread.
        for region in thread.call.func.body:
            if isinstance(region, ParameterLoc):
                value = thread.call.bindings[region.param.name]
                if isinstance(value, SVPlaceholder):
                    length += value.max_length
                else:
                    parrot_assert(
                        isinstance(value, str), f"Value must be str. Got {type(value)}"
                    )
                    length += len(self.tokenizer.tokenize(value, tokenizer_name))
            else:
                parrot_assert(
                    isinstance(region, Constant),
                    f"Invalid region type: {type(region)}.",
                )
                length += len(self.tokenizer.tokenize(region.text, tokenizer_name))

        return length

    def accept_thread(self, thread: "Thread"):
        """Accept a thread to this self."""

        thread.engine = self
        thread_len = self.count_thread_token_nums(thread)
        self.threads.append(thread)
        self.threads_len[thread.unique_id] = thread_len
        self.tokens_num += thread_len

    def remove_thread(self, thread: "Thread"):
        """Remove a thread from this self."""

        # Don't do this! Because hence the thread will be marked as not dispatched.
        # thread.engine = None

        self.threads.remove(thread)
        self.tokens_num -= self.threads_len[thread.unique_id]
        self.threads_len.pop(thread.unique_id)
