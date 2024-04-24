# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


class ParrotError(Exception):
    "Base class for all Parrot exceptions."

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __repr__(self) -> str:
        return f"ParrotError(type={type(self._exc)}, msg={self._exc.args[0]})"


class ParrotOSUserError(ParrotError):
    """This type of error doesn't affect the internal state of OS. It will be passed back
    to the client to handle it."""


class ParrotEngineUserError(ParrotError):
    """This type of error doesn't affect the internal state of Engine. It will be passed back
    to the client to handle it."""


class ParrotOSInternalError(ParrotError):
    """This type of error represents a unrecoverable error in the ParrotOS, which means
    when this error is raised, the ParrotOS will be terminated."""


class ParrotEngineInternalError(ParrotError):
    """This type of error represents a unrecoverable error in the ParrotEngine, which means
    when this error is raised, the ParrotEngine will be terminated."""


def parrot_assert(cond: bool, msg: str):
    if not cond:
        raise ParrotError(AssertionError(msg))
