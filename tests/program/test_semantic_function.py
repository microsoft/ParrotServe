import pytest
import parrot as P

from parrot.program.function import Constant, ParameterLoc, SemanticCall


def test_parse_semantic_function():
    @P.semantic_function()
    def tell_me_a_joke(
        topic: P.Input,
        keyword: P.Input,
        joke: P.Output,
        explanation: P.Output,
    ):
        """Tell me a joke about {{topic}}. The joke must contains the following
        keywords: {{keyword}}. The following is the joke: {{joke}}. And giving a
        short explanation to show that why it is funny. The following is the explanation
        for the joke above: {{explanation}}."""

    expected_body = [
        Constant,
        ParameterLoc,
        Constant,
        ParameterLoc,
        Constant,
        ParameterLoc,
        Constant,
        ParameterLoc,
    ]
    expected_var_is_output = [
        False,
        False,
        True,
        True,
    ]
    assert len(expected_body) == len(tell_me_a_joke.body)
    j = 0
    for i, piece in enumerate(tell_me_a_joke.body):
        assert isinstance(piece, expected_body[i])
        if isinstance(piece, ParameterLoc):
            assert piece.param.is_output == expected_var_is_output[j]
            j += 1


def test_parse_semantic_function_corner_cases():
    @P.semantic_function()
    def single_output(output: P.Output):
        """This is a test for single output: {{output}}"""

    @P.semantic_function()
    def pure_locs(
        input: P.Input,
        output: P.Output,
    ):
        """This is a test for pure locs: {{input}}{{output}}"""

    @P.semantic_function()
    def test_utf8(
        input: P.Input,
        output: P.Output,
    ):
        """This is a test for utf8: {{input}}中文{{output}}"""

    pickled = test_utf8("a").pickle()
    # print(pickled)


def test_call_function():
    @P.semantic_function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    print(test("a", b="b"))


def test_call_function_with_pyobjects():
    @P.semantic_function()
    def test(a: float, b: int, c: list, d: P.Output):
        """This {{b}} is a test {{a}} function {{c}} and {{d}}"""

    print(test.body)

    print(test(23.3, 400, [1, 2, 3, 4]))


def test_wrongly_pass_output_argument():
    # NOTE: output argument can only be passed by name

    @P.semantic_function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    with pytest.raises(ValueError):
        test("a", "b", "c")


def test_serialize_call():
    @P.semantic_function()
    def test(a: P.Input, b: P.Input, c: P.Output):
        """This {{b}} is a test {{a}} function {{c}}"""

    call = test("a", b="b")
    print(call)
    call_pickled = call.pickle()
    # print(call_pickled)
    call_unpickled = SemanticCall.unpickle(call_pickled)
    print(call_unpickled)

    assert call.func.name == call_unpickled.func.name
    assert len(call.func.body) == len(call_unpickled.func.body)
    for i, piece in enumerate(call.func.body):
        assert type(piece) == type(call_unpickled.func.body[i])

    assert len(call.bindings) == len(call_unpickled.bindings)
    for k, v in call.bindings.items():
        assert type(call_unpickled.bindings[k]) == type(v)


if __name__ == "__main__":
    # test_parse_semantic_function()
    test_parse_semantic_function_corner_cases()
    # test_call_function()
    # test_call_function_with_pyobjects()
    # test_wrongly_pass_output_argument()
    # test_serialize_call()
