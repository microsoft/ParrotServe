# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Dict
import time

from parrot.exceptions import parrot_assert
from parrot.utils import get_logger

from .primitive_job import PrimitiveJob, Fill, Generate
from .config import SchedulerConfig


logger = get_logger("Scheduler")


class Scheduler:
    """Scheduler for the backend."""

    def __init__(self, config: SchedulerConfig):
        self.max_batch_size = config.max_batch_size
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_total_tokens = config.max_total_tokens

        self.waiting_jobs: List[PrimitiveJob] = []
        self.running_jobs: List[PrimitiveJob] = []

        self.policy = config.policy

        # Use context id as key. Different jobs with the same context id can't
        # present at the same time.
        self._job_arrival_time: Dict[int, float] = {}

        # pid_tid as key.
        self._thread_arrival_time: Dict[str, float] = {}

    def add_job(self, job: PrimitiveJob):
        """Add a job to the scheduler."""

        self.waiting_jobs.append(job)
        cur_time = time.perf_counter()
        self._job_arrival_time[job.context_id] = cur_time
        key = f"{job.pid}_{job.tid}"
        if key not in self._thread_arrival_time:
            self._thread_arrival_time[key] = cur_time

    def remove_job(self, job: PrimitiveJob):
        """Remove a job from the scheduler."""

        # self.running_jobs.remove(job)
        self._job_arrival_time.pop(job.context_id)
        if job.end_flag:
            key = f"{job.pid}_{job.tid}"
            self._thread_arrival_time.pop(key)

    @property
    def num_running_jobs(self) -> int:
        """Get the number of running jobs."""

        return len(self.running_jobs)

    @property
    def num_total_jobs(self) -> int:
        """Get the number of total jobs."""

        return len(self.waiting_jobs) + len(self.running_jobs)

    @property
    def empty(self) -> bool:
        """Check if the scheduler is empty."""

        # print(f"Waiting: {len(self.waiting_jobs)} Running: {len(self.running_jobs)}")
        # return len(self.waiting_jobs) == 0 and len(self.running_jobs) == 0
        return self.num_total_jobs == 0

    def schedule(self) -> List[PrimitiveJob]:
        """Schedule jobs."""

        if self.policy == "fifo_v1":
            cur_num_jobs = len(self.running_jobs)
            cur_num_batched_tokens = len(
                self.running_jobs
            )  # Note: running jobs must be all Gen jobs.
            cur_total_tokens = 0

            visited_context_ids = (
                set()
            )  # This is to avoid compute the same context multiple times.

            for job in self.running_jobs:
                parrot_assert(
                    isinstance(job, Generate), "Running job must be a Generate job."
                )
                ctx = job.context
                if ctx.context_id not in visited_context_ids:
                    cur_total_tokens += ctx.get_this_context_len()
                    visited_context_ids.add(ctx.context_id)
                parent_ctx = ctx.parent_context
                if parent_ctx and parent_ctx.context_id not in visited_context_ids:
                    cur_total_tokens += parent_ctx.get_this_context_len()
                    visited_context_ids.add(parent_ctx.context_id)

            # print(
            #     f"Scheduling: Waiting: {len(self.waiting_jobs)} Running: {len(self.running_jobs)}"
            # )

            while self.waiting_jobs:
                job = self.waiting_jobs[0]

                job_num_tokens = (
                    1
                    if isinstance(job, Generate) or job.token_ids is None
                    else len(job.token_ids)
                )
                job_total_tokens = job.context.get_this_context_len()
                if (
                    job.context.parent_context
                    and job.context.parent_context.context_id not in visited_context_ids
                ):
                    job_total_tokens += (
                        job.context.parent_context.get_this_context_len()
                    )

                # Constraints
                if cur_num_jobs + 1 > self.max_batch_size:
                    break
                if (
                    cur_num_batched_tokens + job_num_tokens
                    > self.max_num_batched_tokens
                ):
                    break
                if cur_total_tokens + job_total_tokens > self.max_total_tokens:
                    break

                self.running_jobs.append(job)
                if job.start_time == -1:
                    job.start_time = time.perf_counter_ns()
                self.waiting_jobs.pop(0)

                # Update
                cur_num_jobs += 1
                cur_num_batched_tokens += job_num_tokens
                cur_total_tokens += job_total_tokens

            # NOTE(chaofan): Use copy() to avoid list modification.
            ret = self.running_jobs.copy()
            return ret

        # TGI-style scheduling: Fill and Gen jobs are scheduled separately.
        if self.policy == "tgi":
            cur_tokens_sum = 0
            cur_num_jobs = 0
            fill_running_jobs = []

            for job in self.waiting_jobs:
                if not isinstance(job, Fill):
                    continue

                job_num_tokens = len(job.token_ids) if job.token_ids else 0

                if cur_tokens_sum + job_num_tokens > self.max_num_batched_tokens:
                    break

                fill_running_jobs.append(job)
                if job.start_time == -1:
                    job.start_time = time.perf_counter_ns()
                cur_tokens_sum += job_num_tokens
                cur_num_jobs += 1

            if len(fill_running_jobs) > 0:
                # Remove all fill_running_jobs from waiting_jobs.
                self.waiting_jobs = [
                    job for job in self.waiting_jobs if job not in fill_running_jobs
                ]

                # Preempte all running Generation jobs.
                self.waiting_jobs = self.running_jobs + self.waiting_jobs  # FIFO
                self.running_jobs = fill_running_jobs
                return fill_running_jobs.copy()

        cur_num_jobs = len(self.running_jobs)
        cur_num_batched_tokens = len(
            self.running_jobs
        )  # Note: running jobs must be all Gen jobs.

        # print(
        #     f"Scheduling: Waiting: {len(self.waiting_jobs)} Running: {len(self.running_jobs)}"
        # )

        while self.waiting_jobs:
            job = self.waiting_jobs[0]

            job_num_tokens = (
                1
                if isinstance(job, Generate) or job.token_ids is None
                else len(job.token_ids)
            )
            # Constraints
            if cur_num_jobs + 1 > self.max_batch_size:
                break
            if cur_num_batched_tokens + job_num_tokens > self.max_num_batched_tokens:
                break

            self.running_jobs.append(job)
            if job.start_time == -1:
                job.start_time = time.perf_counter_ns()
            self.waiting_jobs.pop(0)

            # Update
            cur_num_jobs += 1
            cur_num_batched_tokens += job_num_tokens

        # Check total tokens constraint and do preemption

        # This is to avoid compute the same context multiple times.
        # TODO(chaofan): Only do this in shared prefix mode.
        # visited_context_ids = set()
        # if ctx.context_id not in visited_context_ids:
        #     cur_total_tokens += ctx.get_this_context_len()
        #     visited_context_ids.add(ctx.context_id)
        # parent_ctx = ctx.parent_context
        # if parent_ctx and parent_ctx.context_id not in visited_context_ids:
        #     cur_total_tokens += parent_ctx.get_this_context_len()
        #     visited_context_ids.add(parent_ctx.context_id)

        # For normal mode, we repeatly count prefix because it's repeated loaded.

        if self.policy == "fifo_v2":
            # HACK(chaofan): App FIFO
            self.running_jobs.sort(key=lambda job: job.pid)
        else:
            self.running_jobs.sort(
                key=lambda job: (
                    self._thread_arrival_time[f"{job.pid}_{job.tid}"],
                    self._job_arrival_time[job.context_id],
                )
            )

        # HACK(chaofan): App FIFO
        # self.running_jobs.sort(key=lambda job: job.pid)

        # print(f"Running jobs: {self.running_jobs}")
        # print(self._thread_arrival_time)

        new_running: List[PrimitiveJob] = []
        cur_total_tokens = 0
        preempted = False
        for job in self.running_jobs:
            if preempted:
                self._preempt(job)
                continue

            # NOTE(chaofan): In shared prefix mode, we should only count the prefix context once.
            job_tokens = job.context.get_context_len()
            if cur_total_tokens + job_tokens > self.max_total_tokens:
                preempted = True
                self._preempt(job)
                continue

            new_running.append(job)
            cur_total_tokens += job_tokens

        self.running_jobs = new_running

        # NOTE(chaofan): Use copy() to avoid list modification.
        ret = self.running_jobs.copy()
        return ret

    def _preempt(self, job):
        self.waiting_jobs.insert(0, job)
        # logger.debug(f"Job {job} preempted.")

    def finish(self):
        """Finish jobs."""

        new_running: List[PrimitiveJob] = []
        for job in self.running_jobs:
            if not job.finish_event.is_set():
                new_running.append(job)
            else:
                self.remove_job(job)
                job.end_time = time.perf_counter_ns()
                logger.debug(
                    f"Job {job} finished. Latency: {(job.end_time - job.start_time) / 1e6} ms"
                )

        self.running_jobs = new_running
