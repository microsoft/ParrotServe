import secrets
from typing import Optional


def create_session_auth() -> str:
    return secrets.token_urlsafe(32)


def verify_session_auth(supplied: Optional[str], expected: str) -> bool:
    if not isinstance(supplied, str):
        return False
    return secrets.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    )
