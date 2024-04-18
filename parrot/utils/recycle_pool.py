# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List
from collections import deque


class RecyclePool:
    def __init__(self, pool_name: str = "pool"):
        self.pool_name = pool_name
        self.allocated_num = 0
        self.cur_max_id = 0
        self.free_ids: deque[int] = deque()
        self.history_max = 0

    def allocate(self) -> int:
        """Fetch an id."""

        self.allocated_num += 1

        if len(self.free_ids) == 0:
            self.cur_max_id += 1
            return self.cur_max_id - 1

        allocated_id = self.free_ids.popleft()  # Pop from left
        self.history_max = max(self.history_max, self.get_allocated_num())
        return allocated_id

    def free(self, id: int) -> int:
        """Free an id."""

        self.allocated_num -= 1

        if id in self.free_ids:
            raise ValueError("The id is already free.")

        self.free_ids.append(id)  # Append to right

    def get_allocated_num(self) -> int:
        """Get the number of allocated ids."""

        return self.allocated_num

    def get_history_max_allocated_num(self) -> int:
        """Get the maximum number of allocated ids."""

        return self.history_max
