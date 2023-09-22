from ..constants import FILL_NO_CHUNK


class ExecutionEngine:
    """Represent execution engines in high-level management."""

    def __init__(self, name: str, host: str, port: int, tokenizer: str):
        # ---------- Basic Config ----------
        self.name = name
        self.host = host
        self.port = port
        self.tokenizer = tokenizer

        # ---------- Monitor Data ----------
        self.num_cached_tokens = 0
        self.cached_tokens_size = 0
        self.num_running_jobs = 0

        # ---------- Controlled Args ----------
        self.fill_chunk_size = FILL_NO_CHUNK

    @property
    def http_address(self) -> str:
        return f"http://{self.host}:{self.port}"
