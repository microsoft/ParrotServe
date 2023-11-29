import pytest
import inspect
import parrot as P

from parrot.program.function import NativeCall


def test_parse_native_function():
    @P.native_function()
    def add(a: P.Input, b: P.Input) -> P.Output:
        return str(int(a) + int(b))

    def add_pyfunc(a: str, b: str) -> str:
        return str(int(a) + int(b))

    print(add.display_signature())
    print(add.inputs)
    print(add.outputs)
    print(inspect.signature(add_pyfunc))


def test_parse_native_function_two_rets():
    @P.native_function()
    def add(a: P.Input, b: P.Input) -> (P.Output, P.Output):
        return str(int(a) + int(b)), str(int(a) - int(b))

    def add_pyfunc(a: str, b: str) -> (str, str):
        return str(int(a) + int(b)), str(int(a) - int(b))

    print(add.display_signature())
    print(add.inputs)
    print(add.outputs)
    print(inspect.signature(add_pyfunc))


def test_call_function():
    @P.native_function()
    def add(a: P.Input, b: P.Input) -> P.Output:
        return str(int(a) + int(b))

    call = add("1", b="2")
    print(call)

    pyfunc = add.get_pyfunc()
    result = pyfunc("1", b="2")
    print(result)


def test_serialize_call():
    @P.native_function()
    def add(a: P.Input, b: P.Input) -> P.Output:
        return str(int(a) + int(b))

    call = add("1", b="2")
    print(call)
    call_pickled = call.pickle()
    # print(call_pickled)
    call_unpickled = NativeCall.unpickle(call_pickled)
    print(call_unpickled)

    assert call.func.name == call_unpickled.func.name
    assert len(call.func.params) == len(call_unpickled.func.params)
    for p1, p2 in zip(call.func.params, call_unpickled.func.params):
        assert p1.name == p2.name
        assert p1.typ == p2.typ

    assert len(call.bindings) == len(call_unpickled.bindings)
    for k, v in call.bindings.items():
        assert type(call_unpickled.bindings[k]) == type(v)

    pyfunc = call_unpickled.func.get_pyfunc()
    ret = pyfunc("1", b="2")
    print(ret)

    pyfunc = call.func.get_pyfunc()
    ret = pyfunc("1", b="2")
    print(ret)


if __name__ == "__main__":
    test_parse_native_function()
    test_parse_native_function_two_rets()
    test_call_function()
    test_serialize_call()
