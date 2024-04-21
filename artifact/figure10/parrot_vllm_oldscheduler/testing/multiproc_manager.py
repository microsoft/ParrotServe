# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from multiprocessing import Process, Lock, Manager, Barrier


class MultiProcessManager:
    """A util to run multi processes and gather the results."""

    def __init__(self):
        self.lock = Lock()
        self.manager = Manager()
        self.data = self.manager.dict()  # id -> return value
        self.jobs = []

        self._counter = 0

    def _proc_wrapper(self, id: int, target, args):
        """A wrapper for the target function."""

        ret = target(*args)
        with self.lock:
            self.data[id] = ret

    def add_proc(self, target, args):
        """Add a process to run."""

        process = Process(target=self._proc_wrapper, args=(self._counter, target, args))
        self.jobs.append(process)
        self._counter += 1

    def run_all(self):
        """Run all processes. Get the results from `self.data`."""

        for job in self.jobs:
            job.start()

        for job in self.jobs:
            job.join()

    def reset(self):
        """Reset the manager."""

        self.data.clear()
        self.jobs.clear()
        self._counter = 0
