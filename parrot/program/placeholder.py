from typing import Optional


class Placeholder:
    """Placeholder for context variables."""

    def __init__(self, name: str, content: Optional[str]):
        self.name = name
        self.content = content

    @property
    def ready(self) -> bool:
        return self.content is not None
