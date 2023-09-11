import logging
import asyncio


def get_logger(log_name: str, log_level: int = logging.DEBUG):
    logger = logging.getLogger(log_name)

    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class RecyclePool:
    def __init__(self, pool_size: int):
        self.flags = [False] * pool_size
        self.recent_free = -1
        self.pool_size = pool_size

    def allocate(self) -> int:
        """Fetch an id."""

        allocated_id = -1

        if self.recent_free >= 0:
            allocated_id = self.recent_free
            self.recent_free = -1
        else:
            for i in range(self.pool_size):
                if not self.flags[i]:
                    allocated_id = i
                    break
        if allocated_id == -1:
            raise ValueError("No free id in the pool.")
        self.flags[allocated_id] = True
        return allocated_id

    def free(self, index: int) -> int:
        """Free an id."""

        if not self.flags[index]:
            raise ValueError("The id is already free.")

        self.flags[index] = False
        self.recent_free = index


def run_new_coro_in_current_loop(coro):
    loop = asyncio.get_running_loop()
    asyncio.run_coroutine_threadsafe(coro, loop)
