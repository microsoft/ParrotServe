class ExecutionEngine:
    """Represent execution engines in high-level management."""

    def __init__(self, name: str, host: str, port: int):
        # ---------- Basic Config ----------
        self.name = name
        self.host = host
        self.port = port

        # ---------- Monitor Data ----------
        self.cached_tokens = 0
        self.running_jobs = 0

    @property
    def http_address(self) -> str:
        return f"http://{self.host}:{self.port}"
