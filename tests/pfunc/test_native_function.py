import pytest
import inspect
from parrot import P

from parrot.frontend.pfunc.function import PyNativeCall


def test_parse_native_function():
    @P.native_function()
    def add(a: P.Input, b: P.Input, c: P.Output):
        ret = str(int(a) + int(b))
        c.set(ret)

    def add_pyfunc(a: str, b: str) -> str:
        return str(int(a) + int(b))

    print(add.display_signature())
    print(add.inputs)
    print(add.outputs)
    print(inspect.signature(add_pyfunc))


def test_call_to_payload():
    @P.native_function()
    def add(a: str, b: str, c: P.Output):
        ret = str(int(a) + int(b))
        c.set(ret)

    call: PyNativeCall = add("1", "2")
    print(call.to_request_payload())


if __name__ == "__main__":
    # test_parse_native_function()
    test_call_to_payload()
