# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from typing import List
from collections import deque


class RecyclePool:
    def __init__(
        self,
        pool_name: str = "pool",
        pool_size: int = 1024,
    ):
        self.pool_name = pool_name
        self.pool_size = pool_size
        self.free_ids: deque[int] = deque(list(range(pool_size)))

    def allocate(self) -> int:
        """Fetch an id."""

        if len(self.free_ids) == 0:
            raise ValueError(f"No free id in the pool: {self.pool_name}.")
        allocated_id = self.free_ids.popleft()  # Pop from left
        return allocated_id

    def free(self, id: int) -> int:
        """Free an id."""

        if id in self.free_ids:
            raise ValueError("The id is already free.")

        self.free_ids.append(id)  # Append to right
