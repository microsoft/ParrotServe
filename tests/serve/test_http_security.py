import asyncio

from starlette.requests import Request

from parrot.serve.http_server import (
    AUTH_TOKEN_ENV,
    engine_heartbeat,
    register_engine,
    require_api_key,
    submit_py_native_call,
)
from parrot.serve.session_auth import create_session_auth, verify_session_auth


def _request(authorization=None):
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    return Request({"type": "http", "headers": headers})


def test_requests_are_rejected_without_server_token(monkeypatch):
    monkeypatch.delenv(AUTH_TOKEN_ENV, raising=False)

    async def call_next(request):
        raise AssertionError("Locked requests must not reach an endpoint.")

    response = asyncio.run(require_api_key(_request(), call_next))

    assert response.status_code == 503


def test_requests_require_valid_bearer_token(monkeypatch):
    monkeypatch.setenv(AUTH_TOKEN_ENV, "test-token")

    async def call_next(request):
        raise AssertionError("Unauthorized requests must not reach an endpoint.")

    response = asyncio.run(require_api_key(_request(), call_next))

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_valid_bearer_token_reaches_endpoint(monkeypatch):
    monkeypatch.setenv(AUTH_TOKEN_ENV, "test-token")
    expected_response = object()

    async def call_next(request):
        return expected_response

    response = asyncio.run(
        require_api_key(_request("Bearer" + " " + "test-token"), call_next)
    )

    assert response is expected_response


def test_python_native_calls_are_disabled():
    response = asyncio.run(submit_py_native_call(_request()))

    assert response.status_code == 410
    assert b"disabled" in response.body


def test_dynamic_engine_registration_is_disabled():
    register_response = asyncio.run(register_engine(_request()))
    heartbeat_response = asyncio.run(engine_heartbeat(_request()))

    assert register_response.status_code == 403
    assert heartbeat_response.status_code == 403


def test_session_credentials_are_unique_and_unpredictable():
    first = create_session_auth()
    second = create_session_auth()

    assert first != second
    assert len(first) >= 32
    assert verify_session_auth(first, first)
    assert not verify_session_auth(second, first)
    assert not verify_session_auth(None, first)
