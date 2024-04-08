# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum
from typing import List, Dict, Optional

from parrot.protocol.internal.runtime_info import EngineRuntimeInfo

# WARN(chaofan): Import from engine package
from parrot.engine.config import EngineConfig

from parrot.constants import (
    ENGINE_TYPE_BUILTIN,
    ENGINE_TYPE_OPENAI,
)
from parrot.exceptions import parrot_assert

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

        # task_id -> upperbound
        self.tasks_num_upperbounds: Dict[int, int] = {}


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
        self.bad_exception: Optional[Exception] = None

        # ---------- Runtime Info ----------

        # NOTE(chaofan): There are two info packages in the runtime info:
        # - Real-time info: The info that changes frequently, like the number of running jobs.
        #                   This info type is sent from engine to OS in heartbeat messages.
        # - Serve-layer info: Info maintained by ServeCore.
        #
        # Synchronization between these two info packages is necessary: by updating the static info
        # when the real-time info changes.

        self._real_time_runtime_info = EngineRuntimeInfo()
        self._serve_layer_runtime_info = ServeLayerRuntimeInfo()

    # ---------- Status Methods ----------

    def mark_bad(self, exception: Exception) -> None:
        self.status = EngineStatus.BAD
        self.bad_exception = exception

    @property
    def is_running(self) -> bool:
        return self.status == EngineStatus.RUNNING

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
    def tokenizer_name(self) -> str:
        return self.model.tokenizer_name

    @property
    def requires_token_ids(self) -> bool:
        return self.model_type == ModelType.TOKEN_ID

    # ---------- For Scheduling ----------

    def get_num_tasks(self) -> int:
        """Return the number of tasks scheduled to this engine."""

        return self._serve_layer_runtime_info.num_tasks

    def get_tokens_num(self) -> int:
        """Return the number of tokens scheduled to this engine."""

        return self._serve_layer_runtime_info.tokens_num

    def get_remain_tokens_capacity(self) -> int:
        """Return the number of tokens that can be scheduled to this engine."""

        return self.config.tokens_capacity - self._serve_layer_runtime_info.tokens_num

    def get_remain_tasks_capacity(self) -> int:
        """Return the number of tasks that can be scheduled to this engine."""

        return self.config.tasks_capacity - self._serve_layer_runtime_info.num_tasks

    def get_tasks_num_upperbound(self) -> int:
        """Return the upperbound of the number of tasks of this engine."""

        return min(
            [self.config.tasks_capacity]
            + list(self._serve_layer_runtime_info.tasks_num_upperbounds.values())
        )

    def update_realtime_runtime_info(self, runtime_info: EngineRuntimeInfo) -> None:
        """Update the real-time runtime info of the engine."""

        self._real_time_runtime_info = runtime_info

    def update_servelayer_runtime_info_add_task(self, task: "CompletionTask") -> None:
        """Update the serve-layer runtime info by a task scheduled to it."""

        parrot_assert(task.is_scheduled, "The task is not scheduled.")

        self._serve_layer_runtime_info.num_tasks += 1

        tokens_num = task.get_token_nums(self.tokenizer_name)
        self._serve_layer_runtime_info.tokens_num += tokens_num

        tasks_num_upperbound = task.schedule_annotation.tasks_num_upperbound
        self._serve_layer_runtime_info.tasks_num_upperbounds[task.task_id] = (
            tasks_num_upperbound
        )

    def update_servelayer_runtime_info_remove_task(
        self, task: "CompletionTask"
    ) -> None:
        """Update the serve-layer runtime info by a task removed from it."""

        parrot_assert(task.is_scheduled, "The task is not scheduled.")

        self._serve_layer_runtime_info.num_tasks -= 1

        tokens_num = task.get_token_nums(self.tokenizer_name)
        self._serve_layer_runtime_info.tokens_num -= tokens_num

        self._serve_layer_runtime_info.tasks_num_upperbounds.pop(task.task_id)

    # ---------- For Profiling ----------

    def get_cache_mem(self) -> float:
        return self._real_time_runtime_info.cache_mem

    def get_num_cached_tokens(self) -> int:
        return self._real_time_runtime_info.num_cached_tokens
