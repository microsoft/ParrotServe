from typing import List
from asyncio import Queue as AsyncQueue

from ..constants import PIPE_END_TOKEN_ID


class TokenPipe:
    """Async generator for receiving tokens."""

    def __init__(self, chunk_num: int):
        self.queue: AsyncQueue[int] = AsyncQueue()
        self.chunk_num = chunk_num

    async def generator(self):
        chunk: List[int] = []

        while True:
            token_id = await self.queue.get()
            if token_id == PIPE_END_TOKEN_ID:
                break

            chunk.append(token_id)

            if len(chunk) >= self.chunk_num:
                yield chunk
                chunk = []

        # Handle the last batch
        if len(chunk) != 0:
            yield chunk
