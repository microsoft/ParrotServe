from typing import List
from .backend_jobs import BackendPrimitiveJob


class Scheduler:
    """Scheduler for the backend."""

    def __init__(self, max_batch_size: int, max_tokens_sum: int):
        self.max_batch_size = max_batch_size
        self.max_tokens_sum = max_tokens_sum

        self.waiting_jobs: List[BackendPrimitiveJob] = []
        self.running_jobs: List[BackendPrimitiveJob] = []

    def add_job(self, job: BackendPrimitiveJob):
        """Add a job to the scheduler."""

        self.waiting_jobs.append(job)

    @property
    def empty(self) -> bool:
        """Check if the scheduler is empty."""

        return len(self.waiting_jobs) == 0 and len(self.running_jobs) == 0

    def schedule(self) -> List[BackendPrimitiveJob]:
        """Schedule jobs."""

        cur_num_jobs = len(self.running_jobs)
        cur_tokens_sum = 0

        for job in self.running_jobs:
            cur_tokens_sum += job.context.get_context_len()

        # print(
        #     f"Scheduling: Waiting: {len(self.waiting_jobs)} Running: {len(self.running_jobs)}"
        # )

        while self.waiting_jobs:
            job = self.waiting_jobs[0]
            job_tokens_num = job.context.get_context_len()

            if cur_tokens_sum + job_tokens_num > self.max_tokens_sum:
                break

            if cur_num_jobs + 1 > self.max_batch_size:
                break

            self.running_jobs.append(job)
            self.waiting_jobs.pop(0)
            cur_tokens_sum += job_tokens_num
            cur_num_jobs += 1

        # NOTE(chaofan): Use copy() to avoid list modification.
        return self.running_jobs.copy()

    def finish(self):
        """Finish jobs."""

        new_running: List[BackendPrimitiveJob] = []
        for job in self.running_jobs:
            if not job.finish_event.is_set():
                new_running.append(job)
        self.running_jobs = new_running
